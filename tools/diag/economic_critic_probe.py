from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src"), str(ROOT)]

import torch

import winner_train as wt
from kaggrl.economic_critic import (
    MONEY_SCALE,
    REGRESSION_TARGETS,
    RISK_TARGETS,
    EconomicCritic,
    build_critic_examples,
    economic_critic_loss,
    examples_to_tensors,
)


def _margin_bucket(margin):
    margin = int(margin)
    if margin <= -50000:
        return "lt50"
    if margin <= -30000:
        return "lt30"
    if margin <= -20000:
        return "lt20"
    return "normal"


def split_games(rows, val_fraction=0.25):
    """Stratify by seat and tail bucket to avoid optimistic validation."""
    groups = {}
    for row in sorted(rows, key=lambda row: (row["seed"], row["seat"])):
        key = (int(row["seat"]), _margin_bucket(row["margin"]))
        groups.setdefault(key, []).append(row)

    train, val = [], []
    for key in sorted(groups):
        group = groups[key]
        if len(group) == 1:
            train.extend(group)
            continue
        val_n = max(1, int(round(len(group) * float(val_fraction))))
        val_n = min(val_n, len(group) - 1)
        # Evenly sample validation members across the ordered group rather
        # than always taking the tail end.
        chosen = set()
        for index in range(val_n):
            pos = int(round((index + 1) * (len(group) + 1) / (val_n + 1))) - 1
            pos = max(0, min(len(group) - 1, pos))
            while pos in chosen and pos + 1 < len(group):
                pos += 1
            chosen.add(pos)
        for index, row in enumerate(group):
            (val if index in chosen else train).append(row)

    # If a rare bucket only has one game, move a representative catastrophe
    # to validation only when the training set still keeps another case from
    # the same severity family.
    for bucket in ("lt50", "lt30", "lt20"):
        if any(_margin_bucket(row["margin"]) == bucket for row in val):
            continue
        candidates = [
            row for row in train if _margin_bucket(row["margin"]) == bucket
        ]
        if len(candidates) >= 2:
            row = candidates[-1]
            train.remove(row)
            val.append(row)

    if not val and train:
        val.append(train.pop())
    return train, val


def collect_examples(rows):
    examples = []
    per_game = []
    for row in rows:
        game_examples = build_critic_examples(row.get("catastrophic_trace") or [])
        examples.extend(game_examples)
        per_game.append(
            {
                "seed": int(row["seed"]),
                "seat": int(row["seat"]),
                "margin": int(row["margin"]),
                "examples": len(game_examples),
            }
        )
    return examples, per_game


def regression_metrics(model, batch):
    model.eval()
    with torch.no_grad():
        out = model(batch["features"])
    pred = out.regression
    target = batch["regression"]
    mask = batch["regression_mask"]
    metrics = {}
    for idx, name in enumerate(REGRESSION_TARGETS):
        valid = mask[:, idx] > 0.5
        if not bool(valid.any()):
            metrics[name] = {"mae": None, "count": 0}
            continue
        mae = (
            (pred[valid, idx] - target[valid, idx]).abs().mean()
            * MONEY_SCALE
        )
        metrics[name] = {
            "mae": float(mae.item()),
            "count": int(valid.sum().item()),
        }
    return metrics, out


def terminal_constant_baseline(train_examples, val_examples):
    if not train_examples or not val_examples:
        return None
    train_terminal = [
        row["regression"][6] for row in train_examples
    ]
    mean = sum(train_terminal) / len(train_terminal)
    mae = sum(
        abs(row["regression"][6] - mean) for row in val_examples
    ) / len(val_examples)
    return float(mae * MONEY_SCALE)


def terminal_simple_baselines(val_examples):
    if not val_examples:
        return {}
    target = [float(row["terminal_margin"]) for row in val_examples]
    cash = [
        float(row.get("cash_margin_baseline", 0))
        for row in val_examples
    ]
    economic = [
        float(row.get("visible_horizon_baseline", 0))
        for row in val_examples
    ]
    return {
        "cash_margin_mae": sum(
            abs(a - b) for a, b in zip(target, cash)
        ) / len(target),
        "visible_horizon_mae": sum(
            abs(a - b) for a, b in zip(target, economic)
        ) / len(target),
    }


def _binary_metrics(probs, truth, threshold):
    pred = probs >= float(threshold)
    tp = int((truth & pred).sum().item())
    fp = int((~truth & pred).sum().item())
    fn = int((truth & ~pred).sum().item())
    tn = int((~truth & ~pred).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (
        2.0 * precision * recall / max(1e-9, precision + recall)
        if tp > 0 else 0.0
    )
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def risk_metrics(output, batch):
    probs = torch.sigmoid(output.risk_logits)
    target = batch["risk"]
    result = {}
    thresholds = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)
    for idx, name in enumerate(RISK_TARGETS):
        truth = target[:, idx] > 0.5
        sweep = [
            _binary_metrics(probs[:, idx], truth, threshold)
            for threshold in thresholds
        ]
        default = next(
            row for row in sweep if abs(row["threshold"] - 0.50) < 1e-9
        )
        result[name] = {
            "positive": int(truth.sum().item()),
            **{key: value for key, value in default.items() if key != "threshold"},
            "mean_probability": float(probs[:, idx].mean().item()),
            "threshold_sweep": sweep,
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--parent", default=str(ROOT / "assets" / "parent_promoted_v2.pt"))
    ap.add_argument("--opponent", default=str(ROOT / "assets" / "v51_main.py"))
    ap.add_argument("--seed-start", type=int, default=23238520)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--iteration", type=int, default=1205)
    ap.add_argument("--temperature", type=float, default=1.00242003614173)
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--output", default="/tmp/economic_critic_probe.pt")
    args = ap.parse_args()

    random.seed(260923)
    torch.manual_seed(260923)
    os.environ["FARMOS_CATASTROPHIC_MARGIN"] = "999999999"
    wt.CATASTROPHIC_MARGIN = 10**9
    seeds = list(range(args.seed_start, args.seed_start + args.games))
    rows = wt.exact_games(
        args.parent,
        args.snapshot,
        args.opponent,
        seeds,
        args.workers,
        True,
        args.temperature,
        1.0,
        2.3,
        24,
        args.iteration,
        both_seats=True,
        skill_cutover_step=672,
        skill_keep_penalty=2.75,
        skill_shadow_start_step=0,
        skill_confidence_threshold=0.70,
    )
    train_rows, val_rows = split_games(rows)
    train_examples, train_games = collect_examples(train_rows)
    val_examples, val_games = collect_examples(val_rows)
    if not train_examples or not val_examples:
        raise RuntimeError("economic critic probe has empty train/val examples")

    device = torch.device("cpu")
    train_batch = examples_to_tensors(train_examples, device=device)
    val_batch = examples_to_tensors(val_examples, device=device)
    model = EconomicCritic(hidden_dim=args.hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )

    best_state = None
    best_val = math.inf
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = economic_critic_loss(model, train_batch)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_losses = economic_critic_loss(model, val_batch)
        score = float(val_losses["loss"].item())
        if score < best_val:
            best_val = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    train_reg, train_out = regression_metrics(model, train_batch)
    val_reg, val_out = regression_metrics(model, val_batch)
    report = {
        "games": len(rows),
        "train_games": train_games,
        "val_games": val_games,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "constant_terminal_mae": terminal_constant_baseline(
            train_examples, val_examples
        ),
        "simple_terminal_baselines": terminal_simple_baselines(
            val_examples
        ),
        "train_regression": train_reg,
        "val_regression": val_reg,
        "train_risk": risk_metrics(train_out, train_batch),
        "val_risk": risk_metrics(val_out, val_batch),
        "rollout": {
            "mean_margin": sum(row["margin"] for row in rows) / len(rows),
            "worst_margin": min(row["margin"] for row in rows),
            "catastrophes": sum(row["margin"] <= -30000 for row in rows),
        },
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "farmos_economic_critic_probe_v1",
            "model_state": model.state_dict(),
            "hidden_dim": args.hidden,
            "report": report,
        },
        target,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
