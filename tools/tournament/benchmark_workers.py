from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src"), str(ROOT)]

import winner_train as wt


def parse_workers(text: str):
    return [
        max(1, int(part.strip()))
        for part in text.split(",")
        if part.strip()
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--snapshot",
        default=str(ROOT / "assets" / "tournament_actor_iter1205.pt"),
    )
    ap.add_argument(
        "--parent",
        default=str(ROOT / "assets" / "parent_promoted_v2.pt"),
    )
    ap.add_argument(
        "--opponent",
        default=str(ROOT / "assets" / "v51_main.py"),
    )
    ap.add_argument(
        "--gate",
        default=str(ROOT / "assets" / "act_keep_gate_sweep_best.pt"),
    )
    ap.add_argument("--workers", default="4,8,12,16")
    ap.add_argument("--seed-start", type=int, default=23239500)
    ap.add_argument("--seed-count", type=int, default=4)
    ap.add_argument("--iteration", type=int, default=1205)
    args = ap.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    results = []
    for workers in parse_workers(args.workers):
        started = time.perf_counter()
        rows = wt.exact_games(
            args.parent,
            args.snapshot,
            args.opponent,
            seeds,
            workers,
            True,
            1.00242003614173,
            1.0,
            2.3,
            24,
            args.iteration,
            both_seats=True,
            skill_cutover_step=672,
            skill_keep_penalty=2.75,
            skill_shadow_start_step=0,
            skill_confidence_threshold=0.70,
            runtime_options={
                "act_keep_gate_path": args.gate,
                "act_keep_threshold": 0.50,
            },
        )
        elapsed = time.perf_counter() - started
        margins = [int(row["margin"]) for row in rows]
        games = len(rows)
        result = {
            "workers": workers,
            "games": games,
            "elapsed_sec": elapsed,
            "games_per_minute": games / max(elapsed, 1e-9) * 60.0,
            "mean_margin": statistics.fmean(margins),
            "worst_margin": min(margins),
        }
        results.append(result)
        print(
            f"[WORKERS] n={workers:02d} games={games} "
            f"elapsed={elapsed:.1f}s "
            f"gpm={result['games_per_minute']:.2f} "
            f"mean={result['mean_margin']:+.1f}",
            flush=True,
        )

    best = max(results, key=lambda row: row["games_per_minute"])
    print(
        json.dumps(
            {
                "cpu_count": os.cpu_count(),
                "best_workers": best["workers"],
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
