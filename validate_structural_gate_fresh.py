import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT), str(ROOT / "src")]

from kaggle_environments import make
from continuous_runtime import ContinuousRuntime
from rollout.v4_hybrid_agent import V4HybridRolloutAgent


def run(task):
    seed, seat, gate = task
    rt = ContinuousRuntime(
        ROOT / "assets" / "parent_promoted_v2.pt",
        ROOT / "output" / "causal_v4_1_actor.pt",
        stochastic=False,
        temperature=0.90,
        residual_scale=1.0,
        base_keep_bias=2.3,
        decision_every=24,
        enable_structural_route_gate=bool(gate),
        seed=int(seed) * 13 + int(seat),
    )
    learner = V4HybridRolloutAgent(
        option_policy=rt,
        min_option_confidence=0.0,
        enable_market_race_ordering=True,
        enable_idle_weed_labor=True,
        enable_structural_route_gate=bool(gate),
    )
    opp = str(ROOT / "assets" / "v51_main.py")
    agents = [learner, opp] if seat == 0 else [opp, learner]
    env = make(
        "kaggriculture",
        configuration={"seed": int(seed), "episodeSteps": 720},
        debug=False,
    )
    env.run(agents)
    farms = env.steps[-1][0].observation.farms
    own = int(farms[seat].money)
    rival = int(farms[1 - seat].money)
    rec144 = next(
        (r for r in rt.records if int(r["step"]) == 144),
        None,
    )
    return {
        "seed": int(seed),
        "seat": int(seat),
        "gate": bool(gate),
        "margin": own - rival,
        "own": own,
        "v51": rival,
        "win": int(own > rival),
        "base_route_144": (
            int(rec144["base_route_id"]) if rec144 else None
        ),
        "chosen_route_144": (
            int(rec144["chosen_route_id"]) if rec144 else None
        ),
    }


def metrics(rows):
    margins = [int(r["margin"]) for r in rows]
    return {
        "games": len(rows),
        "wins": sum(int(r["win"]) for r in rows),
        "mean_margin": sum(margins) / len(margins),
        "median_margin": sorted(margins)[len(margins) // 2],
        "min_margin": min(margins),
        "max_margin": max(margins),
    }
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=22993000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--output",
        default=str(ROOT / "output" / "structural_gate_fresh_v4_2.json"),
    )
    args = ap.parse_args()
    games = [
        (args.seed_start + i, seat)
        for i in range(args.seeds)
        for seat in (0, 1)
    ]
    output = Path(args.output)
    data = {
        "schema": "farmos_structural_gate_fresh_validation_v4_2",
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "baseline": [],
        "structural_gate": [],
    }
    ctx = mp.get_context("spawn")
    for name, gate in (("baseline", False), ("structural_gate", True)):
        tasks = [(seed, seat, gate) for seed, seat in games]
        rows = []
        with ctx.Pool(processes=min(args.workers, len(tasks))) as pool:
            for i, row in enumerate(
                pool.imap_unordered(run, tasks, chunksize=1), 1
            ):
                rows.append(row)
                rows.sort(key=lambda r: (r["seed"], r["seat"]))
                data[name] = rows
                output.write_text(
                    json.dumps(data, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"[{name}] {i}/{len(tasks)} "
                    f"seed={row['seed']} seat={row['seat']} "
                    f"base144={row['base_route_144']} "
                    f"margin={row['margin']:+d}",
                    flush=True,
                )
        data[name + "_metrics"] = metrics(rows)
        output.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(name.upper(), data[name + "_metrics"], flush=True)

    baseline = {
        (r["seed"], r["seat"]): r for r in data["baseline"]
    }
    gate_rows = {
        (r["seed"], r["seat"]): r for r in data["structural_gate"]
    }
    paired = []
    for key in sorted(baseline):
        b = baseline[key]
        g = gate_rows[key]
        paired.append({
            "seed": key[0],
            "seat": key[1],
            "base_route_144": b["base_route_144"],
            "baseline_margin": b["margin"],
            "gate_margin": g["margin"],
            "gain": int(g["margin"]) - int(b["margin"]),
        })
    gains = [p["gain"] for p in paired]
    data["paired"] = paired
    data["delta"] = {
        "mean_gain": sum(gains) / len(gains),
        "positive": sum(x > 0 for x in gains),
        "neutral": sum(x == 0 for x in gains),
        "negative": sum(x < 0 for x in gains),
        "baseline_wins": data["baseline_metrics"]["wins"],
        "gate_wins": data["structural_gate_metrics"]["wins"],
    }
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("DELTA", data["delta"], flush=True)
    for row in sorted(paired, key=lambda r: r["gain"], reverse=True):
        print("[PAIR]", row, flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
