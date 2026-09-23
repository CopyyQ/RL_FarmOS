from __future__ import annotations
import torch
from torch import nn

from continuous_runtime import ContinuousActor, init_actor, load_actor_state_compatible, load_parent
from diag_act_keep_gate import (
    ROOT, RUN, CKPT, TRAIN, VAL,
    load_records, micro_features, binary_metrics, current_flat_predictions,
)
from winner_train import skill_teacher_records_to_batch


def build_gate(kind, dim):
    if kind == "linear":
        return nn.Linear(dim, 2)
    if kind == "mlp":
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )
    raise ValueError(kind)


def train_gate(kind, keep_weight, z_train, y_train, z_val, y_val, device):
    gate = build_gate(kind, z_train.shape[-1]).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=1e-3, weight_decay=1e-4)
    weights = torch.tensor([float(keep_weight), 1.0], device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(20260923)
    best = None
    best_state = None
    for epoch in range(25):
        perm = torch.randperm(z_train.shape[0], generator=gen, device=device)
        for start in range(0, z_train.shape[0], 1024):
            idx = perm[start:start + 1024]
            loss = nn.functional.cross_entropy(
                gate(z_train[idx]), y_train[idx], weight=weights
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.inference_mode():
            pred = gate(z_val).argmax(dim=-1)
        m = binary_metrics(pred, y_val)
        score = min(m["KEEP"]["f1"], m["ACT"]["f1"])
        if best is None or score > best:
            best = score
            best_state = {
                k: v.detach().cpu().clone() for k, v in gate.state_dict().items()
            }
            best_metrics = m
            best_epoch = epoch + 1
    return best, best_epoch, best_metrics, best_state


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload, _ = load_parent(ROOT / "assets" / "parent_promoted_v2.pt", device=device)
    actor = ContinuousActor(
        int(payload["hidden_dim"]), int(payload["clock_dim"]),
        len(payload["route_ids"]), len(payload["market_modes"]),
    ).to(device)
    init_actor(actor)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    load_actor_state_compatible(actor, ckpt["actor_state"])
    actor.eval()

    train = skill_teacher_records_to_batch(
        load_records(TRAIN), int(payload["hidden_dim"]), int(payload["clock_dim"]), device
    )
    val = skill_teacher_records_to_batch(
        load_records(VAL), int(payload["hidden_dim"]), int(payload["clock_dim"]), device
    )
    y_train = (train["target"] != 0).long()
    y_val = (val["target"] != 0).long()
    z_train = micro_features(actor, train)
    z_val = micro_features(actor, val)

    flat = binary_metrics(current_flat_predictions(actor, val), y_val)
    print("BASE", flat, flush=True)
    results = []
    for kind in ("linear", "mlp"):
        for keep_weight in (1.0, 1.5, 2.0, 3.0, 4.0):
            best, epoch, metrics, state = train_gate(
                kind, keep_weight, z_train, y_train, z_val, y_val, device
            )
            row = {
                "kind": kind,
                "keep_weight": keep_weight,
                "epoch": epoch,
                "score": best,
                "metrics": metrics,
                "state": state,
            }
            results.append(row)
            print(
                "CFG", kind, "KW", keep_weight, "E", epoch,
                "ACC", round(metrics["accuracy"], 4),
                "KEEP_F1", round(metrics["KEEP"]["f1"], 4),
                "KEEP_P", round(metrics["KEEP"]["precision"], 4),
                "KEEP_R", round(metrics["KEEP"]["recall"], 4),
                "ACT_F1", round(metrics["ACT"]["f1"], 4),
                flush=True,
            )

    winner = max(results, key=lambda x: x["score"])
    out = RUN / "act_keep_gate_sweep_best.pt"
    torch.save(
        {
            "schema": "farmos_v46_act_keep_gate_sweep_v1",
            "checkpoint": str(CKPT),
            "kind": winner["kind"],
            "keep_weight": winner["keep_weight"],
            "epoch": winner["epoch"],
            "metrics": winner["metrics"],
            "state_dict": winner["state"],
            "input_dim": int(z_train.shape[-1]),
        },
        out,
    )
    print(
        "BEST", winner["kind"], winner["keep_weight"], winner["epoch"],
        winner["metrics"], flush=True,
    )
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
