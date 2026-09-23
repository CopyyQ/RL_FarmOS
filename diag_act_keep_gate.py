from __future__ import annotations
from pathlib import Path
import torch
from torch import nn

from continuous_runtime import (
    ContinuousActor,
    init_actor,
    load_actor_state_compatible,
    load_parent,
)
from winner_train import (
    MICRO_TASKS,
    bitmask_to_mask,
    skill_teacher_records_to_batch,
)

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "v46_econ_shadow"
CKPT = RUN / "winner_v45_skill_stage_safe.pt"
TRAIN = RUN / "skill_teacher_train_v46.pt"
VAL = RUN / "skill_teacher_val_v46.pt"


def load_records(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return [{"skill_teacher_records": list(payload.get("records", []))}]
def micro_features(actor, batch, chunk=2048):
    zs = []
    actor.eval()
    with torch.inference_mode():
        n = int(batch["hidden"].shape[0])
        for start in range(0, n, chunk):
            idx = slice(start, min(n, start + chunk))
            x = torch.cat(
                (
                    batch["hidden"][idx],
                    batch["clock"][idx],
                    batch["local"][idx],
                    batch["economic"][idx],
                    batch["opponent"][idx],
                ),
                dim=-1,
            )
            z = actor.micro_trunk(x)
            pc = batch["plant_context"][idx]
            z = z + torch.tanh(actor.micro_plant_context(pc))
            zs.append(z.detach())
    return torch.cat(zs, dim=0)


def binary_metrics(pred, target):
    out = {}
    for cls, name in ((0, "KEEP"), (1, "ACT")):
        true = target == cls
        guess = pred == cls
        tp = int((true & guess).sum().item())
        fp = int((~true & guess).sum().item())
        fn = int((true & ~guess).sum().item())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        out[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(true.sum().item()),
        }
    out["accuracy"] = float((pred == target).float().mean().item())
    return out


def current_flat_predictions(actor, batch, keep_penalty=2.75, chunk=2048):
    preds = []
    with torch.inference_mode():
        n = int(batch["hidden"].shape[0])
        for start in range(0, n, chunk):
            idx = slice(start, min(n, start + chunk))
            out = actor.micro_forward(
                batch["hidden"][idx],
                batch["clock"][idx],
                batch["local"][idx],
                economic=batch["economic"][idx],
                opponent=batch["opponent"][idx],
                plant_context=batch["plant_context"][idx],
            )
            logits = out["task"]
            mask = bitmask_to_mask(batch["task_bits"][idx], len(MICRO_TASKS))
            nonkeep_valid = mask[:, 1:].any(dim=-1)
            logits = logits.clone()
            logits[nonkeep_valid, 0] -= keep_penalty
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            preds.append((logits.argmax(dim=-1) != 0).long())
    return torch.cat(preds)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload, _ = load_parent(ROOT / "assets" / "parent_promoted_v2.pt", device=device)
    actor = ContinuousActor(
        int(payload["hidden_dim"]),
        int(payload["clock_dim"]),
        len(payload["route_ids"]),
        len(payload["market_modes"]),
    ).to(device)
    init_actor(actor)
    state = torch.load(CKPT, map_location=device, weights_only=False)
    load_actor_state_compatible(actor, state["actor_state"])
    actor.eval()

    train = skill_teacher_records_to_batch(
        load_records(TRAIN),
        int(payload["hidden_dim"]),
        int(payload["clock_dim"]),
        device,
    )
    val = skill_teacher_records_to_batch(
        load_records(VAL),
        int(payload["hidden_dim"]),
        int(payload["clock_dim"]),
        device,
    )
    train_y = (train["target"] != 0).long()
    val_y = (val["target"] != 0).long()

    z_train = micro_features(actor, train)
    z_val = micro_features(actor, val)
    gate = nn.Linear(z_train.shape[-1], 2).to(device)
    nn.init.zeros_(gate.weight)
    nn.init.zeros_(gate.bias)

    counts = torch.bincount(train_y, minlength=2).float()
    weights = counts.sum() / (2.0 * counts.clamp_min(1.0))
    opt = torch.optim.AdamW(gate.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator(device=device)
    generator.manual_seed(20260923)
    batch_size = 1024

    for epoch in range(20):
        perm = torch.randperm(z_train.shape[0], generator=generator, device=device)
        losses = []
        for start in range(0, z_train.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            logits = gate(z_train[idx])
            loss = nn.functional.cross_entropy(logits, train_y[idx], weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if epoch in {0, 1, 4, 9, 19}:
            with torch.inference_mode():
                pred = gate(z_val).argmax(dim=-1)
            m = binary_metrics(pred, val_y)
            print(
                "EPOCH", epoch + 1,
                "LOSS", round(sum(losses) / max(1, len(losses)), 5),
                "VAL_ACC", round(m["accuracy"], 4),
                "KEEP_F1", round(m["KEEP"]["f1"], 4),
                "KEEP_P", round(m["KEEP"]["precision"], 4),
                "KEEP_R", round(m["KEEP"]["recall"], 4),
                "ACT_F1", round(m["ACT"]["f1"], 4),
                flush=True,
            )

    flat_pred = current_flat_predictions(actor, val)
    flat = binary_metrics(flat_pred, val_y)
    with torch.inference_mode():
        gate_pred = gate(z_val).argmax(dim=-1)
    gated = binary_metrics(gate_pred, val_y)
    print("CURRENT_FLAT", flat, flush=True)
    print("ACT_KEEP_GATE", gated, flush=True)

    out = RUN / "act_keep_gate_probe.pt"
    torch.save(
        {
            "schema": "farmos_v46_act_keep_gate_probe_v1",
            "checkpoint": str(CKPT),
            "state_dict": gate.state_dict(),
            "input_dim": int(z_train.shape[-1]),
            "metrics": gated,
        },
        out,
    )
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
