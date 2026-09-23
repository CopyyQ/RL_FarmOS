from __future__ import annotations

from collections import Counter

from tools.diag.economic_candidate_shop_coverage import make_reference_observation
from tools.diag.economic_candidate_shop_screen_v2 import _animal_payback
from tools.tournament.generate_economic_candidates_v2 import FAMILY_COUNTS, build_candidates
from kaggrl.v45_economics import strategy_snapshot


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


if __name__ == "__main__":
    test_v2_generator_has_expected_raw_pool_and_all_families()
    test_v2_animal_payback_is_finite_and_semantic()
    print("ECONOMIC_PORTFOLIO_V2_TESTS_OK")
