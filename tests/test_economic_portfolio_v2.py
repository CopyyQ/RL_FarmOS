from __future__ import annotations

from collections import Counter

from tools.diag.economic_candidate_shop_coverage import make_reference_observation
from tools.diag.economic_candidate_shop_screen_v2 import (
    _animal_payback,
    _apply_supply_regime,
    _crop_target,
    _screen_snapshot,
)
from tools.tournament.generate_economic_candidates_v2 import FAMILY_COUNTS, build_candidates
from kaggrl.v45_economics import strategy_snapshot
from kaggrl.v4_farm_supervisor import apply_portfolio_switch_overlay


def test_v2_generator_has_expected_raw_pool_and_all_families():
    candidates = build_candidates()
    assert len(candidates) == 1 + sum(FAMILY_COUNTS.values()) == 301
    counts = Counter(c["family"] for c in candidates)
    assert counts["official_baseline"] == 1
    for family, expected in FAMILY_COUNTS.items():
        assert counts[family] == expected


def test_v2_animal_payback_is_finite_and_semantic():
    obs = make_reference_observation({"YARN_STORE": 4}, step=288)
    snap = strategy_snapshot(
        obs,
        projected_crop_value=True,
        projected_crop_horizon_extra_days=4.0,
    )
    sheep = _animal_payback(snap, obs, "SHEEP")
    assert sheep["cycles"] >= 1
    assert sheep["cost"] > 500.0
    assert sheep["revenue"] >= 0.0
    assert sheep["roi"] >= 0.0
    assert -1.0 <= sheep["margin"] < 100.0


def test_v2_runtime_rewrites_animal_target_without_changing_slot_or_quantity():
    obs = make_reference_observation({"YARN_STORE": 4}, step=288)
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["BUY_ANIMAL", "COW", 2]],
    }
    out, meta = apply_portfolio_switch_overlay(
        obs,
        action,
        activation_step=72,
        source_mode="none",
        objective="animal",
        forced_animal="SHEEP",
        animal_min_roi=0.0,
        animal_payback_margin=-1.0,
    )
    assert len(out["market"]) == len(action["market"]) == 1
    assert out["market"][0][0] == "BUY_ANIMAL"
    assert out["market"][0][1] == "SHEEP"
    assert out["market"][0][2] == 2
    assert meta["animal_switches"] == 1


def test_v2_animal_objective_does_not_convert_seed_action_type():
    obs = make_reference_observation({"YARN_STORE": 4}, step=288)
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["BUY_SEED", "WHEAT", 2]],
    }
    out, meta = apply_portfolio_switch_overlay(
        obs,
        action,
        activation_step=72,
        source_mode="none",
        objective="animal",
        forced_animal="SHEEP",
        animal_min_roi=0.0,
        animal_payback_margin=-1.0,
    )
    assert out is action
    assert out["market"] == action["market"]
    assert not meta["applied"]


def test_v2_glut_family_is_noop_neutral_but_activates_on_realized_glut():
    candidate = next(
        c for c in build_candidates()
        if c["id"] == "v2_B_glut_to_scarcity_001"
    )
    options = candidate["runtime_options"]
    horizon = float(options["portfolio_horizon_extra_days"])
    base = make_reference_observation({"BAKERY": 1}, step=72)
    neutral = _screen_snapshot(
        base, horizon, cache_animal_payback=False
    )
    assert _crop_target(neutral, options, "TOMATO") == "TOMATO"

    glut_obs = _apply_supply_regime(
        base,
        {
            "id": "realized_glut_tomato",
            "crop": "TOMATO",
            "inventory_delta": 5000,
        },
    )
    glut = _screen_snapshot(
        glut_obs, horizon, cache_animal_payback=False
    )
    assert _crop_target(glut, options, "TOMATO") == "STRAWBERRY"


if __name__ == "__main__":
    test_v2_generator_has_expected_raw_pool_and_all_families()
    test_v2_animal_payback_is_finite_and_semantic()
    test_v2_runtime_rewrites_animal_target_without_changing_slot_or_quantity()
    test_v2_animal_objective_does_not_convert_seed_action_type()
    test_v2_glut_family_is_noop_neutral_but_activates_on_realized_glut()
    print("ECONOMIC_PORTFOLIO_V2_TESTS_OK")
