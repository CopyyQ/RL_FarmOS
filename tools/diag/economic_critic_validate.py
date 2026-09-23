from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src"), str(ROOT)]

import torch

import winner_train as wt
from kaggrl.economic_critic import (
    ECON_CRITIC_FEATURE_DIM,
    EconomicCritic,
    build_critic_examples,
    examples_to_tensors,
)
from tools.diag.economic_critic_probe import (
    regression_metrics,
    risk_metrics,
    terminal_simple_baselines,
)


def main():
    artifact = Path("/tmp/economic_critic_probe32.pt")
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    model = EconomicCritic(hidden_dim=int(payload["hidden_dim"]))
    model.load_state_dict(payload["model_state"])
    model.eval()

    os.environ["FARMOS_CATASTROPHIC_MARGIN"] = "999999999"
    wt.CATASTROPHIC_MARGIN = 10**9
    seeds = [23238522, 23238526, 23238530, 23238532, 23238533]
    rows = wt.exact_games(
        str(ROOT / "assets" / "parent_promoted_v2.pt"),
        "/home/plab/Desktop/Nguyen_Anh_Quyet_PLAB/AI/gpu_v51_winner_v4/"
        "runs/v46_econ_shadow/_rollout_actor_v4.pt",
        str(ROOT / "assets" / "v51_main.py"),
        seeds,
        8,
        True,
        1.00242003614173,
        1.0,
        2.3,
        24,
        1205,
        both_seats=True,
        skill_cutover_step=672,
        skill_keep_penalty=2.75,
        skill_shadow_start_step=0,
        skill_confidence_threshold=0.70,
    )
    examples = []
    games = []
    for row in rows:
        game_examples = build_critic_examples(
            row.get("catastrophic_trace") or []
        )
        examples.extend(game_examples)
        games.append({
            "seed": int(row["seed"]),
            "seat": int(row["seat"]),
            "margin": int(row["margin"]),
            "examples": len(game_examples),
        })

    batch = examples_to_tensors(examples)
    regression, output = regression_metrics(model, batch)
    report = {
        "games": games,
        "examples": len(examples),
        "feature_dim": ECON_CRITIC_FEATURE_DIM,
        "simple_terminal_baselines": terminal_simple_baselines(examples),
        "regression": regression,
        "risk": risk_metrics(output, batch),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
