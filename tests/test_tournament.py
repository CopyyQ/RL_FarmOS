from pathlib import Path
import tempfile

from tools.tournament.generate_runtime_candidates import build_candidates
from tools.tournament.run import (
    ResultCache,
    paired_deltas,
    resolve_stage_seeds,
    score_candidate,
    summarize,
)


def _row(seed, seat, margin, win=0):
    return {
        "seed": seed,
        "seat": seat,
        "margin": margin,
        "win": win,
        "plant_deaths": 0,
        "animal_escapes": 0,
        "invalid_ops": 0,
        "harvested_units": 0,
        "sold_units": 0,
    }


def test_default_manifest_has_100_and_one_baseline():
    candidates = build_candidates()
    assert len(candidates) == 100
    baseline = [row for row in candidates if row.get("baseline")]
    assert len(baseline) == 1
    row = baseline[0]
    assert row["rollout"]["skill_cutover_step"] == 672
    assert row["rollout"]["skill_confidence_threshold"] == 0.70
    assert row["runtime_options"]["act_keep_threshold"] == 0.50


def test_stage_seed_range():
    assert resolve_stage_seeds(
        {"name": "x", "seed_start": 100, "seed_count": 3}
    ) == [100, 101, 102]


def test_summary_and_pairing():
    baseline = [_row(1, 0, -1000), _row(1, 1, -2000)]
    candidate = [_row(1, 0, 0, 1), _row(1, 1, -1000)]
    bm = summarize(baseline)
    cm = summarize(candidate)
    paired = paired_deltas(baseline, candidate)
    assert bm["mean_margin"] == -1500
    assert cm["mean_margin"] == -500
    assert paired["mean_delta"] == 1000
    assert paired["improved"] == 2


def test_hard_tail_gate_rejects_regression():
    baseline = summarize([_row(1, 0, -10000), _row(1, 1, -10000)])
    candidate = summarize([_row(1, 0, -30000), _row(1, 1, -30000)])
    paired = {
        "games": 2,
        "mean_delta": -20000,
        "median_delta": -20000,
        "improved": 0,
        "equal": 0,
        "worse": 2,
        "best_delta": -20000,
        "worst_delta": -20000,
    }
    _, failures = score_candidate(
        baseline,
        candidate,
        paired,
        {
            "hard_cvar_floor": -3000,
            "hard_worst_floor": -7000,
            "hard_mean_floor": -2000,
        },
    )
    assert failures


def test_cache_replaces_same_seed_seat():
    with tempfile.TemporaryDirectory() as td:
        cache = ResultCache(Path(td), "run")
        cache.append("candidate", [_row(1, 0, -1000)])
        cache.append("candidate", [_row(1, 0, 500)])
        rows = cache.load("candidate")
        assert rows[(1, 0)]["margin"] == 500


if __name__ == "__main__":
    test_default_manifest_has_100_and_one_baseline()
    test_stage_seed_range()
    test_summary_and_pairing()
    test_hard_tail_gate_rejects_regression()
    test_cache_replaces_same_seed_seat()
    print("TOURNAMENT_TESTS_OK")
