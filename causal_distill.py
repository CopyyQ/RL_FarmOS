import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from continuous_runtime import (
    ContinuousActor,
    HORIZONS,
    init_actor,
    load_actor_state_compatible,
    load_parent,
)
from winner_train import bitmask_to_mask, save_snapshot


def _softmax(values, temperature):
    x = np.asarray(values, dtype=np.float64)
    x = (x - x.max()) / max(1e-6, float(temperature))
    x = np.exp(np.clip(x, -60.0, 0.0))
    return x / x.sum()


def load_source_actor(actor, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("actor_state")
    if not isinstance(state, dict):
        raise ValueError(f"{checkpoint}: actor_state missing")
    info = load_actor_state_compatible(actor, state)
    print(
        f"ACTOR_LOAD matched={info['matched']} "
        f"missing={len(info['missing'])}",
        flush=True,
    )
    return payload


def index_records(memory_path):
    payload = torch.load(memory_path, map_location="cpu", weights_only=False)
    index = {}
    for game in payload.get("games", []):
        key0 = (int(game["seed"]), int(game["seat"]))
        for record in game.get("records", []):
            index[(key0[0], key0[1], int(record["step"]))] = record
    return index


def build_groups(causal_path, record_index, payload, device, tau, min_spread):
    causal = json.loads(Path(causal_path).read_text(encoding="utf-8"))
    market_modes = tuple(payload["market_modes"])
    market_to_id = {str(x): i for i, x in enumerate(market_modes)}
    horizon_to_id = {int(x): i for i, x in enumerate(HORIZONS)}

    grouped = {}
    for row in causal.get("rows", []):
        key = (
            int(row["seed"]),
            int(row["seat"]),
            str(row["kind"]),
            int(row["step"]),
        )
        grouped.setdefault(key, []).append(row)

    market_groups = []
    horizon_groups = []
    for (seed, seat, kind, step), rows in grouped.items():
        record = record_index.get((seed, seat, step))
        state_row = rows[0]
        margins = [float(x["margin"]) for x in rows]
        spread = max(margins) - min(margins)
        if spread < float(min_spread):
            continue

        # Prefer hidden/clock captured from the exact replay that produced the
        # counterfactual return. Fallback keeps older causal files readable.
        if (
            "hidden_f16_hex" in state_row
            and "clock_f16_hex" in state_row
        ):
            hidden_hex = str(state_row["hidden_f16_hex"])
            clock_hex = str(state_row["clock_f16_hex"])
            if any(
                str(row.get("hidden_f16_hex", hidden_hex)) != hidden_hex
                or str(row.get("clock_f16_hex", clock_hex)) != clock_hex
                for row in rows
            ):
                # The target action cannot affect the state before it is taken;
                # differing state captures mean the causal group is invalid.
                continue
            hidden = np.frombuffer(
                bytes.fromhex(hidden_hex), np.float16
            ).astype(np.float32)
            clock = np.frombuffer(
                bytes.fromhex(clock_hex), np.float16
            ).astype(np.float32)
            base = int(state_row["base_route_class"])
            route = int(state_row["route_action"])
            market = int(state_row["market_action"])
            market_mask_bits = int(state_row["market_mask_bits"])
        else:
            if record is None:
                continue
            hidden = np.frombuffer(
                record["hidden_f16"], np.float16
            ).astype(np.float32)
            clock = np.frombuffer(
                record["clock_f16"], np.float16
            ).astype(np.float32)
            base = int(record["base_route_class"])
            route = int(record["route_action"])
            market = int(record["market_action"])
            market_mask_bits = int(record["market_mask_bits"])

        weight = 1.0 + min(10.0, spread / 1000.0)

        if kind == "market":
            target = np.zeros(len(market_modes), dtype=np.float32)
            probs = _softmax(margins, tau)
            for row, prob in zip(rows, probs):
                idx = market_to_id.get(str(row["choice"]))
                if idx is not None:
                    target[idx] += float(prob)
            total = float(target.sum())
            if total <= 0:
                continue
            target /= total
            market_groups.append(
                (hidden, clock, base, route, market_mask_bits,
                 target, weight, spread, seed, seat, step)
            )
        elif kind == "horizon":
            target = np.zeros(len(HORIZONS), dtype=np.float32)
            probs = _softmax(margins, tau)
            for row, prob in zip(rows, probs):
                idx = horizon_to_id.get(int(row["choice"]))
                if idx is not None:
                    target[idx] += float(prob)
            total = float(target.sum())
            if total <= 0:
                continue
            target /= total
            horizon_groups.append(
                (hidden, clock, base, route, market,
                 target, weight, spread, seed, seat, step)
            )
    return market_groups, horizon_groups
def _tensor(x, device, dtype):
    return torch.as_tensor(x, device=device, dtype=dtype)


def train_market(actor, groups, optimizer, market_count, device, epochs):
    if not groups:
        return {"loss": 0.0, "top1": 0.0, "groups": 0}
    hidden = _tensor(np.stack([g[0] for g in groups]), device, torch.float32)
    clock = _tensor(np.stack([g[1] for g in groups]), device, torch.float32)
    base = _tensor([g[2] for g in groups], device, torch.long)
    route = _tensor([g[3] for g in groups], device, torch.long)
    bits = _tensor([g[4] for g in groups], device, torch.long)
    target = _tensor(np.stack([g[5] for g in groups]), device, torch.float32)
    weight = _tensor([g[6] for g in groups], device, torch.float32)
    mask = bitmask_to_mask(bits, market_count)
    target = target * mask.to(target.dtype)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)

    stats = []
    actor.train()
    for _ in range(max(1, int(epochs))):
        out = actor(hidden, clock, base, route_action=route)
        logits = out["market"].masked_fill(
            ~mask, torch.finfo(out["market"].dtype).min
        )
        logp = torch.log_softmax(logits, -1)
        per = -(target * logp).sum(-1)
        w = weight / weight.mean().clamp_min(1e-6)
        loss = (per * w).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]],
            0.5,
        )
        optimizer.step()
        with torch.no_grad():
            top1 = (logits.argmax(-1) == target.argmax(-1)).float().mean()
        stats.append((float(loss.detach()), float(top1)))
    actor.eval()
    return {
        "loss": sum(x[0] for x in stats) / len(stats),
        "top1": sum(x[1] for x in stats) / len(stats),
        "groups": len(groups),
    }


def train_horizon(actor, groups, optimizer, device, epochs):
    if not groups:
        return {"loss": 0.0, "top1": 0.0, "groups": 0}
    hidden = _tensor(np.stack([g[0] for g in groups]), device, torch.float32)
    clock = _tensor(np.stack([g[1] for g in groups]), device, torch.float32)
    base = _tensor([g[2] for g in groups], device, torch.long)
    route = _tensor([g[3] for g in groups], device, torch.long)
    market = _tensor([g[4] for g in groups], device, torch.long)
    target = _tensor(np.stack([g[5] for g in groups]), device, torch.float32)
    weight = _tensor([g[6] for g in groups], device, torch.float32)

    stats = []
    actor.train()
    for _ in range(max(1, int(epochs))):
        out = actor(
            hidden,
            clock,
            base,
            route_action=route,
            market_action=market,
        )
        logits = out["horizon"]
        logp = torch.log_softmax(logits, -1)
        per = -(target * logp).sum(-1)
        w = weight / weight.mean().clamp_min(1e-6)
        loss = (per * w).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]],
            0.5,
        )
        optimizer.step()
        with torch.no_grad():
            top1 = (logits.argmax(-1) == target.argmax(-1)).float().mean()
        stats.append((float(loss.detach()), float(top1)))
    actor.eval()
    return {
        "loss": sum(x[0] for x in stats) / len(stats),
        "top1": sum(x[1] for x in stats) / len(stats),
        "groups": len(groups),
    }
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--causal",
        default=str(ROOT / "output" / "causal_refine_best_winner_v4_1.json"),
    )
    ap.add_argument(
        "--memory",
        default=str(ROOT / "output" / "winner_memory_v4.pt"),
    )
    ap.add_argument(
        "--checkpoint",
        default=str(ROOT / "output" / "winner_v4_latest.pt"),
    )
    ap.add_argument(
        "--parent",
        default=str(ROOT / "assets" / "parent_promoted_v2.pt"),
    )
    ap.add_argument(
        "--output",
        default=str(ROOT / "output" / "causal_v4_1_actor.pt"),
    )
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--return-temperature", type=float, default=600.0)
    ap.add_argument("--min-spread", type=float, default=100.0)
    args = ap.parse_args()

    if not Path(args.causal).exists():
        raise FileNotFoundError(args.causal)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload, _ = load_parent(args.parent, device=device)
    actor = ContinuousActor(
        int(payload["hidden_dim"]),
        int(payload["clock_dim"]),
        len(payload["route_ids"]),
        len(payload["market_modes"]),
    ).to(device)
    init_actor(actor)
    source = load_source_actor(actor, args.checkpoint)

    # Freeze route/trunk/value. Causal distillation only changes the economic
    # choice heads and their conditional adapters.
    for parameter in actor.parameters():
        parameter.requires_grad_(False)

    # Preserve the global V4 policy. Causal refinement only learns
    # conditional residual corrections for states where exact counterfactual
    # games show a material return spread.
    market_names = (
        "route_action_embedding.weight",
    )
    horizon_names = (
        "market_action_embedding.weight",
    )
    market_params = []
    horizon_params = []
    for name, parameter in actor.named_parameters():
        if (
            name in market_names
            or name.startswith("market_condition.")
        ):
            parameter.requires_grad_(True)
            market_params.append(parameter)
        if (
            name in horizon_names
            or name.startswith("horizon_condition.")
        ):
            parameter.requires_grad_(True)
            horizon_params.append(parameter)

    record_index = index_records(Path(args.memory))
    market_groups, horizon_groups = build_groups(
        args.causal,
        record_index,
        payload,
        device,
        args.return_temperature,
        args.min_spread,
    )
    print(
        f"CAUSAL_GROUPS market={len(market_groups)} "
        f"horizon={len(horizon_groups)}",
        flush=True,
    )
    market_opt = torch.optim.AdamW(
        market_params, lr=args.learning_rate, weight_decay=0.0
    )
    horizon_opt = torch.optim.AdamW(
        horizon_params, lr=args.learning_rate, weight_decay=0.0
    )

    for block in range(0, max(1, args.epochs), 10):
        ep = min(10, max(1, args.epochs) - block)
        m = train_market(
            actor,
            market_groups,
            market_opt,
            len(payload["market_modes"]),
            device,
            ep,
        )
        h = train_horizon(
            actor,
            horizon_groups,
            horizon_opt,
            device,
            ep,
        )
        print(
            f"[CAUSAL-DISTILL] epoch={block+ep:03d} "
            f"market_loss={m['loss']:.4f} top1={100*m['top1']:.1f}% "
            f"horizon_loss={h['loss']:.4f} top1={100*h['top1']:.1f}%",
            flush=True,
        )

    save_snapshot(
        actor,
        Path(args.output),
        {
            "schema": "farmos_v51_causal_distill_v4_1",
            "source_checkpoint": str(args.checkpoint),
            "causal_file": str(args.causal),
            "market_groups": len(market_groups),
            "horizon_groups": len(horizon_groups),
            "return_temperature": args.return_temperature,
            "min_spread": args.min_spread,
        },
    )
    print(f"CAUSAL_ACTOR_SAVED {args.output}", flush=True)


if __name__ == "__main__":
    main()
