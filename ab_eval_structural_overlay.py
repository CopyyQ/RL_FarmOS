import json
import multiprocessing as mp
import sys
from pathlib import Path

import torch

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
    seed, seat, mode, snapshot = task
    if mode == "baseline":
        structural = None
    elif mode == "cow123_144_240":
        structural = WindowRoute(144, 240, 123)
    elif mode == "cow105_144_216":
        structural = WindowRoute(144, 216, 105)
    else:
        raise ValueError(mode)

    rt = ContinuousRuntime(
        ROOT / "assets" / "parent_promoted_v2.pt",
        snapshot,
        stochastic=False,
        temperature=0.90,
        residual_scale=1.0,
        base_keep_bias=2.3,
        decision_every=24,
        seed=int(seed) * 17 + int(seat),
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
    opponent = str(ROOT / "assets" / "v51_main.py")
    agents = [learner, opponent] if int(seat) == 0 else [opponent, learner]
    env = make(
        "kaggriculture",
        configuration={"seed": int(seed), "episodeSteps": 720},
        debug=False,
    )
    env.run(agents)
    if len(env.steps) != 720:
        raise RuntimeError((seed, seat, len(env.steps)))
    farms = env.steps[-1][0].observation.farms
    own = int(farms[int(seat)].money)
    rival = int(farms[1 - int(seat)].money)
    return {
        "seed": int(seed),
        "seat": int(seat),
        "mode": mode,
        "own": own,
        "v51": rival,
        "margin": own - rival,
        "win": int(own > rival),
    }


def metrics(rows):
    margins = [int(x["margin"]) for x in rows]
    return {
        "games": len(rows),
        "wins": sum(int(x["win"]) for x in rows),
        "mean_margin": sum(margins) / len(margins),
        "min_margin": min(margins),
        "max_margin": max(margins),
    }


def main():
    memory = torch.load(
        ROOT / "output" / "winner_memory_v4.pt",
        map_location="cpu",
        weights_only=False,
    )
    winners = sorted(
        [g for g in memory["games"] if int(g["margin"]) > 0],
        key=lambda g: int(g["margin"]),
        reverse=True,
    )[:3]
    known_games = [(int(g["seed"]), int(g["seat"])) for g in winners]
    held_games = [
        (seed, seat)
        for seed in range(22990000, 22990004)
        for seat in (0, 1)
    ]
    all_games = known_games + held_games
    modes = ("baseline", "cow123_144_240", "cow105_144_216")
    snapshot = ROOT / "output" / "causal_v4_1_actor.pt"

    result = {}
    ctx = mp.get_context("spawn")
    for mode in modes:
        tasks = [(s, seat, mode, snapshot) for s, seat in all_games]
        with ctx.Pool(processes=4) as pool:
            rows = list(pool.imap_unordered(run, tasks, chunksize=1))
        known = [
            r for r in rows
            if (r["seed"], r["seat"]) in set(known_games)
        ]
        held = [
            r for r in rows
            if (r["seed"], r["seat"]) not in set(known_games)
        ]
        result[mode] = {
            "known": sorted(known, key=lambda r: (r["seed"], r["seat"])),
            "known_metrics": metrics(known),
            "heldout": sorted(held, key=lambda r: (r["seed"], r["seat"])),
            "heldout_metrics": metrics(held),
        }
        print(
            f"[{mode}] known={result[mode]['known_metrics']} "
            f"heldout={result[mode]['heldout_metrics']}",
            flush=True,
        )

    baseline_held = result["baseline"]["heldout_metrics"]["mean_margin"]
    baseline_known = result["baseline"]["known_metrics"]["mean_margin"]
    for mode in modes[1:]:
        print(
            f"[DELTA] {mode} "
            f"known={result[mode]['known_metrics']['mean_margin'] - baseline_known:+.1f} "
            f"heldout={result[mode]['heldout_metrics']['mean_margin'] - baseline_held:+.1f}",
            flush=True,
        )

    out = ROOT / "output" / "structural_overlay_ab_v4_2.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
