from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20260924
FAMILY_COUNTS = {
    "A_wheat_to_scarcity": 40,
    "B_glut_to_scarcity": 40,
    "C_nonbest_to_best_roi": 40,
    "D_any_to_best_guarded": 40,
    "E_crop_diversification": 30,
    "F_shop_regime_specialist": 30,
    "G_cow_milk": 20,
    "H_sheep_wool": 20,
    "I_goose_egg": 20,
    "J_crop_animal_hybrid": 20,
}

RATIOS = (1.04, 1.08, 1.12, 1.20, 1.35, 1.60)
RESERVE_DAYS = (1.0, 2.0, 3.0, 4.0, 6.0)
ACTIVATION_STEPS = (72, 144, 216, 288, 360)
HORIZON_EXTRA_DAYS = (0.0, 2.0, 4.0, 6.0, 10.0)
UNDERSUPPLY = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)
SHOP_DEMAND = (0.0, 1.0, 2.0, 3.0)
ANIMAL_MIN_ROI = (0.70, 0.85, 1.00, 1.15, 1.30)
ANIMAL_PAYBACK_MARGIN = (0.0, 0.10, 0.20, 0.35)
DIVERSITY_PENALTY = (0.0, 0.10, 0.20, 0.35)


def _base_options(rng: random.Random):
    return {
        "enable_portfolio_switch": True,
        "portfolio_crop_improvement_ratio": rng.choice(RATIOS),
        "portfolio_feed_reserve_days": rng.choice(RESERVE_DAYS),
        "portfolio_activation_step": rng.choice(ACTIVATION_STEPS),
        "portfolio_horizon_extra_days": rng.choice(HORIZON_EXTRA_DAYS),
        "portfolio_min_undersupply_ratio": rng.choice(UNDERSUPPLY),
        "portfolio_min_shop_demand": rng.choice(SHOP_DEMAND),
        "portfolio_animal_min_roi": rng.choice(ANIMAL_MIN_ROI),
        "portfolio_animal_payback_margin": rng.choice(ANIMAL_PAYBACK_MARGIN),
        "portfolio_diversity_penalty": rng.choice(DIVERSITY_PENALTY),
    }


def _family_options(family: str, rng: random.Random):
    o = _base_options(rng)
    if family == "A_wheat_to_scarcity":
        o.update(portfolio_source_mode="wheat", portfolio_objective="scarcity_crop")
    elif family == "B_glut_to_scarcity":
        o.update(portfolio_source_mode="glutted", portfolio_objective="scarcity_crop")
    elif family == "C_nonbest_to_best_roi":
        o.update(portfolio_source_mode="nonbest", portfolio_objective="best_crop_roi")
    elif family == "D_any_to_best_guarded":
        o.update(portfolio_source_mode="any", portfolio_objective="best_crop_roi")
        o["portfolio_crop_improvement_ratio"] = max(1.12, o["portfolio_crop_improvement_ratio"])
    elif family == "E_crop_diversification":
        o.update(portfolio_source_mode="any", portfolio_objective="diversified_crop")
    elif family == "F_shop_regime_specialist":
        o.update(portfolio_source_mode="any", portfolio_objective="shop_specialist")
    elif family == "G_cow_milk":
        o.update(portfolio_source_mode="none", portfolio_objective="animal", portfolio_forced_animal="COW")
    elif family == "H_sheep_wool":
        o.update(portfolio_source_mode="none", portfolio_objective="animal", portfolio_forced_animal="SHEEP")
    elif family == "I_goose_egg":
        o.update(portfolio_source_mode="none", portfolio_objective="animal", portfolio_forced_animal="GOOSE")
    elif family == "J_crop_animal_hybrid":
        o.update(portfolio_source_mode="any", portfolio_objective="hybrid")
    else:
        raise ValueError(family)
    return o


def build_candidates():
    rng = random.Random(SEED)
    candidates = [{
        "id": "baseline_official",
        "baseline": True,
        "family": "official_baseline",
        "runtime_options": {"enable_portfolio_switch": False},
    }]
    seen = set()
    for family, count in FAMILY_COUNTS.items():
        made = 0
        attempts = 0
        while made < count:
            attempts += 1
            if attempts > count * 100:
                raise RuntimeError(f"cannot fill family {family}")
            options = _family_options(family, rng)
            key = (family, json.dumps(options, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            made += 1
            candidates.append({
                "id": f"v2_{family}_{made:03d}",
                "baseline": False,
                "family": family,
                "runtime_options": options,
            })
    return candidates


def main():
    candidates = build_candidates()
    payload = {
        "schema": "farmos_candidate_manifest_v2_raw",
        "seed": SEED,
        "description": "Raw multi-family economic portfolio candidates. Must pass shop-space behavioral screening before game tournament.",
        "family_counts": FAMILY_COUNTS,
        "raw_candidate_count": len(candidates),
        "candidates": candidates,
    }
    target = Path("configs/tournament/economic_v2_raw.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {target} candidates={len(candidates)} families={len(FAMILY_COUNTS)}")


if __name__ == "__main__":
    main()
