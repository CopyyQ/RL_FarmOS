from __future__ import annotations
from pathlib import Path

import torch

import winner_train as w

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "assets" / "parent_promoted_v2.pt"
SNAP = ROOT / "runs" / "v46_econ_shadow" / "failed_eval_1120_snapshot.pt"
OPP = ROOT / "assets" / "v51_main.py"


def main():
    seeds = list(range(22990000, 22990004))
    rows = w.exact_games(
        PARENT,
        SNAP,
        OPP,
        seeds,
        8,
        False,
        1.0,
        1.0,
        2.3,
        24,
        1120,
        both_seats=True,
        skill_cutover_step=696,
        skill_keep_penalty=2.75,
        skill_shadow_start_step=0,
        skill_confidence_threshold=0.70,
    )
    failures = []
    for row in rows:
        dense = row.get("dense_stats") or {}
        stats = row.get("skill_stats") or {}
        item = {
            "seed": int(row["seed"]),
            "seat": int(row["seat"]),
            "margin": int(row["margin"]),
            "plant_deaths": int(dense.get("plant_deaths", 0)),
            "animal_escapes": int(dense.get("animal_escapes", 0)),
            "skill_samples": int(stats.get("samples", 0)),
            "skill_proposals": int(stats.get("nonkeep_proposals", 0)),
            "skill_executed": int(stats.get("executed_sampled", 0)),
            "skill_trace": list(row.get("skill_exec_trace") or []),
        }
        print(
            "GAME", item["seed"], item["seat"],
            "margin", item["margin"],
            "plant_deaths", item["plant_deaths"],
            "escapes", item["animal_escapes"],
            "samples", item["skill_samples"],
            "proposals", item["skill_proposals"],
            "executed", item["skill_executed"],
            "trace", len(item["skill_trace"]),
            flush=True,
        )
        if item["animal_escapes"] > 0:
            failures.append(item)

    out = ROOT / "runs" / "v46_econ_shadow" / "eval1120_escape_failures.pt"
    torch.save(
        {
            "schema": "farmos_v46_eval1120_escape_failures_v1",
            "snapshot": str(SNAP),
            "failures": failures,
        },
        out,
    )
    print("FAILURES", len(failures), flush=True)
    for item in failures:
        print(
            "FAIL", item["seed"], item["seat"],
            "margin", item["margin"],
            "escapes", item["animal_escapes"],
            flush=True,
        )
        for trace in item["skill_trace"][-20:]:
            print(" TRACE", trace, flush=True)
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
