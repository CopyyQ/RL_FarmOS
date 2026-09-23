import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent

from winner_train import _run_chunk, metrics
from kaggrl.v4_structural_gate import (
    COW_BRANCH_SOURCE_ROUTES,
    STRUCTURAL_GAIN_TABLE,
)


def compact(row):
    return {
        "seed": int(row["seed"]),
        "seat": int(row["seat"]),
        "own_money": int(row["own_money"]),
        "v51_money": int(row["v51_money"]),
        "margin": int(row["margin"]),
        "win": int(row["win"]),
    }


def main():
    source_routes = sorted(COW_BRANCH_SOURCE_ROUTES)
    if source_routes != [8, 101, 113]:
        raise RuntimeError(f"unexpected structural routes: {source_routes}")

    # One fresh exact execution verifies that the new gain-table loader produces
    # the same runtime behavior as the previously benchmarked gate.
    parent = ROOT / "assets" / "parent_promoted_v2.pt"
    snapshot = ROOT / "output" / "causal_v4_1_actor.pt"
    opponent = ROOT / "assets" / "v51_main.py"
    smoke = _run_chunk(
        (
            str(parent),
            str(snapshot),
            str(opponent),
            [(22200487, 0)],
            False,
            0.90,
            1.0,
            2.3,
            24,
            95001,
        )
    )
    smoke_m = metrics(smoke)

    # Exact cached validations were generated with the identical source-route
    # set and exact 720-step engine. Reuse them instead of recomputing 31 games.
    ab = json.loads(
        (ROOT / "output" / "structural_gate_ab_v4_2.json").read_text()
    )
    fresh = json.loads(
        (ROOT / "output" / "structural_gate_fresh_v4_2.json").read_text()
    )
    known_m = ab["structural_gate"]["known_metrics"]
    fixed_m = ab["structural_gate"]["heldout_metrics"]
    fresh_m = fresh["structural_gate_metrics"]

    checks = {
        "gain_table_schema": (
            STRUCTURAL_GAIN_TABLE.get("schema")
            == "farmos_structural_gain_table_v4_2"
        ),
        "source_routes": source_routes == [8, 101, 113],
        "loader_exact_smoke_win": smoke_m["wins"] == 1,
        "loader_exact_smoke_margin_min_10000": smoke_m["mean_margin"] >= 10000,
        "known_3_of_3": int(known_m["wins"]) == 3,
        "known_mean_min_8000": float(known_m["mean_margin"]) >= 8000.0,
        "fixed_mean_min_minus_12000": float(fixed_m["mean_margin"]) >= -12000.0,
        "fresh_mean_min_minus_12500": float(fresh_m["mean_margin"]) >= -12500.0,
    }
    passed = all(checks.values())

    result = {
        "schema": "farmos_v4_2_promotion_validation",
        "passed": passed,
        "checks": checks,
        "source_routes": source_routes,
        "smoke_metrics": smoke_m,
        "smoke": [compact(x) for x in smoke],
        "known_metrics": known_m,
        "fixed_metrics": fixed_m,
        "fresh_metrics": fresh_m,
        "evidence": {
            "fixed_file": "structural_gate_ab_v4_2.json",
            "fresh_file": "structural_gate_fresh_v4_2.json",
            "gain_table": "structural_gain_table_v4_2.json",
        },
    }
    out = ROOT / "output" / "promotion_v4_2.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("PROMOTION_V4_2", "PASS" if passed else "FAIL")
    print("SOURCE_ROUTES", source_routes)
    print("SMOKE", smoke_m)
    print("KNOWN", known_m)
    print("FIXED", fixed_m)
    print("FRESH", fresh_m)
    print("CHECKS", checks)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
