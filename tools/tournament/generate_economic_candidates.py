from __future__ import annotations

import itertools
import json
from pathlib import Path
import random

SEED = 20260923

# Family V1 is deliberately narrow: surplus WHEAT -> demanded/scarce crop.
# Broad "any crop" switching was falsified by smoke testing because it converted
# STRAWBERRY/CARROT back into WHEAT and caused severe regressions.
RATIOS = (1.10, 1.20, 1.30, 1.40, 1.60)
RESERVE_DAYS = (1.0, 2.0, 3.0, 4.0)
ACTIVATION_STEPS = (72, 144, 216, 288)
HORIZON_EXTRA_DAYS = (0.0, 2.0, 4.0, 6.0)
UNDERSUPPLY = (0.0, 0.10, 0.25, 0.50)
SHOP_DEMAND = (1.0, 2.0, 3.0)
SOURCE_MODE = "wheat"


def _candidate_id(values):
    ratio, reserve, activation, horizon, undersupply, demand = values
    return (
        f"econ_r{ratio:.2f}_f{reserve:.0f}_a{activation}_"
        f"h{horizon:.0f}_u{undersupply:.2f}_d{demand:.0f}_wheat"
    )


def build_candidates():
    baseline = {
        "id": "baseline_official",
        "baseline": True,
        "family": "official_baseline",
        "runtime_options": {
            "enable_portfolio_switch": False,
        },
    }
    space = list(
        itertools.product(
            RATIOS,
            RESERVE_DAYS,
            ACTIVATION_STEPS,
            HORIZON_EXTRA_DAYS,
            UNDERSUPPLY,
            SHOP_DEMAND,
        )
    )
    rng = random.Random(SEED)
    rng.shuffle(space)

    # Anchors include the two strongest safe smoke candidates and boundaries.
    anchors = [
        # Smoke winner: +14k/game on catastrophic+normal pair, no normal change.
        (1.60, 2.0, 144, 4.0, 0.50, 2.0),
        # Second safe smoke arm: +13.6k/game.
        (1.30, 3.0, 216, 2.0, 0.25, 2.0),
        (1.10, 1.0, 72, 6.0, 0.0, 1.0),
        (1.20, 2.0, 144, 4.0, 0.10, 1.0),
        (1.30, 2.0, 144, 4.0, 0.25, 2.0),
        (1.40, 2.0, 216, 4.0, 0.25, 2.0),
        (1.60, 4.0, 288, 0.0, 0.50, 3.0),
        (1.40, 3.0, 216, 6.0, 0.10, 1.0),
    ]

    chosen = []
    seen = set()
    for values in anchors + space:
        if values in seen:
            continue
        seen.add(values)
        chosen.append(values)
        if len(chosen) == 99:
            break

    candidates = [baseline]
    for values in chosen:
        ratio, reserve, activation, horizon, undersupply, demand = values
        candidates.append(
            {
                "id": _candidate_id(values),
                "baseline": False,
                "family": "forward_wheat_to_scarcity_v1",
                "runtime_options": {
                    "enable_portfolio_switch": True,
                    "portfolio_crop_improvement_ratio": float(ratio),
                    "portfolio_feed_reserve_days": float(reserve),
                    "portfolio_activation_step": int(activation),
                    "portfolio_horizon_extra_days": float(horizon),
                    "portfolio_min_undersupply_ratio": float(undersupply),
                    "portfolio_min_shop_demand": float(demand),
                    "portfolio_source_mode": SOURCE_MODE,
                },
            }
        )
    if len(candidates) != 100:
        raise RuntimeError(f"expected 100 candidates, got {len(candidates)}")
    return candidates


def main():
    payload = {
        "schema": "farmos_candidate_manifest_v1",
        "description": (
            "100-candidate safe forward economic crop-portfolio search. "
            "Candidate 0 is the official baseline. The remaining 99 candidates "
            "only rewrite surplus WHEAT investment into a demanded/scarce crop."
        ),
        "base": {
            "temperature": 1.00242003614173,
            "residual_scale": 1.0,
            "base_keep_bias": 2.3,
            "decision_every": 24,
            "skill_cutover_step": 672,
            "skill_keep_penalty": 2.75,
            "skill_shadow_start_step": 0,
            "skill_confidence_threshold": 0.70,
        },
        "candidates": build_candidates(),
    }
    target = Path("configs/tournament/economic100.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {target} candidates={len(payload['candidates'])}")


if __name__ == "__main__":
    main()
