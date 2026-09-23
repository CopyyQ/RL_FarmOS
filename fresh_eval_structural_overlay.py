import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT), str(ROOT / "src")]

from kaggle_environments import make
from kaggrl.clock import resolve_clock
from continuous_runtime import ContinuousRuntime
from rollout.v4_hybrid_agent import V4HybridRolloutAgent


class WindowRoute:
    def __init__(self, start, end, route_id):
        self.start = int(start)
        self.end = int(end)
        self.route_id = int(route_id)

    def __call__(self, observation, configuration, selected_route, context):
        step = int(resolve_clock(observation, configuration).step)
        if self.start <= step < self.end:
            return self.route_id
        return None


def run(task):
    seed, seat, mode = task
    if mode == "baseline":
        structural = None
    elif mode == "cow105_144_216":
        structural = WindowRoute(144, 216, 105)
    elif mode == "cow123_144_240":
        structural = WindowRoute(144, 240, 123)
    else:
        raise ValueError(mode)

    rt = ContinuousRuntime(
        ROOT / "assets" / "parent_promoted_v2.pt",
        ROOT / "output" / "causal_v4_1_actor.pt",
        stochastic=False,
        temperature=0.90,
        residual_scale=1.0,
        base_keep_bias=2.3,
        decision_every=24,
        seed=int(seed) * 31 + int(seat),
    )
    learner = V4HybridRolloutAgent(
        option_policy=rt,
        structural_route_policy=structural,
        min_option_confidence=0.0,
        enable_market_race_ordering=True,
        enable_idle_weed_labor=True,
        enable_late_hire_pruning=False,
        allow_all_routes=False,
    )
    opp = str(ROOT / "assets" / "v51_main.py")
    env = make(
        "kaggriculture",
        configuration={"seed": int(seed), "episodeSteps": 720},
        debug=False,
    )
    env.run([learner, opp] if seat == 0 else [opp, learner])
    farms = env.steps[-1][0].observation.farms
    own = int(farms[seat].money)
    rival = int(farms[1 - seat].money)
    return {
        "seed": seed, "seat": seat, "mode": mode,
        "own": own, "v51": rival,
        "margin": own - rival, "win": int(own > rival),
    }


def metrics(rows):
    margins = [x["margin"] for x in rows]
    return {
        "games": len(rows),
        "wins": sum(x["win"] for x in rows),
        "win_rate": sum(x["win"] for x in rows) / len(rows),
        "mean_margin": sum(margins) / len(margins),
        "median_margin": sorted(margins)[len(margins)//2],
        "min_margin": min(margins),
        "max_margin": max(margins),
    }


def main():
    seeds = list(range(22990100, 22990108))
    games = [(seed, seat) for seed in seeds for seat in (0, 1)]
    modes = ("baseline", "cow105_144_216", "cow123_144_240")
    result = {}
    ctx = mp.get_context("spawn")
    for mode in modes:
        tasks = [(seed, seat, mode) for seed, seat in games]
        with ctx.Pool(processes=4) as pool:
            rows = list(pool.imap_unordered(run, tasks, chunksize=1))
        rows.sort(key=lambda x: (x["seed"], x["seat"]))
        result[mode] = {"metrics": metrics(rows), "rows": rows}
        print(f"[{mode}] {result[mode]['metrics']}", flush=True)

    base = result["baseline"]["metrics"]["mean_margin"]
    for mode in modes[1:]:
        print(
            f"[FRESH-DELTA] {mode} "
            f"{result[mode]['metrics']['mean_margin'] - base:+.1f}",
            flush=True,
        )
    out = ROOT / "output" / "structural_overlay_fresh16_v4_2.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
