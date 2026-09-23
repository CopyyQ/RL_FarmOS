from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src"), str(ROOT)]

import torch

import winner_train as wt
from kaggrl.economic_critic import (
    MONEY_SCALE,
    EconomicCritic,
    encode_economic_snapshot,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--iteration", type=int, default=1205)
    ap.add_argument("--temperature", type=float, default=1.00242003614173)
    args = ap.parse_args()

    os.environ["FARMOS_CATASTROPHIC_MARGIN"] = "999999999"
    wt.CATASTROPHIC_MARGIN = 10**9

    rows = wt.exact_games(
        str(ROOT / "assets" / "parent_promoted_v2.pt"),
        args.snapshot,
        str(ROOT / "assets" / "v51_main.py"),
        [args.seed],
        1,
        True,
        args.temperature,
        1.0,
        2.3,
        24,
        args.iteration,
        both_seats=False,
        skill_cutover_step=672,
        skill_keep_penalty=2.75,
        skill_shadow_start_step=0,
        skill_confidence_threshold=0.70,
    )
    row = rows[0]
    payload = torch.load(args.critic, map_location="cpu", weights_only=False)
    model = EconomicCritic(hidden_dim=int(payload["hidden_dim"]))
    model.load_state_dict(payload["model_state"])
    model.eval()

    print(
        "FINAL",
        row["margin"],
        "SEAT",
        row["seat"],
        "SEED",
        row["seed"],
    )
    first_alert = {0: None, 1: None, 2: None}
    first_prealert = {0: None, 1: None, 2: None}
    last_day = None

    for trace_row in row["catastrophic_trace"]:
        economic = trace_row.get("economic")
        if not economic:
            continue
        features = torch.tensor(
            [encode_economic_snapshot(economic)],
            dtype=torch.float32,
        )
        with torch.no_grad():
            output = model(features)
            terminal = float(output.regression[0, 6].item() * MONEY_SCALE)
            risks = torch.sigmoid(output.risk_logits[0]).tolist()

        step = int(trace_row["step"])
        day = int(trace_row["day"])
        for idx, probability in enumerate(risks):
            if first_prealert[idx] is None and probability >= 0.25:
                first_prealert[idx] = (step, day, probability)
            if first_alert[idx] is None and probability >= 0.50:
                first_alert[idx] = (step, day, probability)

        if day != last_day and day >= 16:
            last_day = day
            print(
                "DAY",
                day,
                "STEP",
                step,
                "CASH",
                int(economic["cash_margin"]),
                "ECON",
                int(economic["visible_horizon_advantage"]),
                "PRED_TERM",
                round(terminal),
                "RISK20/30/50",
                "/".join(f"{p:.3f}" for p in risks),
            )

    print("PREALERT25", first_prealert)
    print("ALERT50", first_alert)


if __name__ == "__main__":
    main()
