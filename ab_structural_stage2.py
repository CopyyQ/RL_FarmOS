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
from kaggrl.v4_structural_gate import COW_BRANCH_SOURCE_ROUTES


class TwoStageStructuralPolicy:
    def __init__(self):
        self.initial_base = None

    def __call__(self, observation, configuration, selected_route, context):
        step = int(resolve_clock(observation, configuration).step)
        if step == 144 and self.initial_base is None:
            self.initial_base = int(context.base_route_id)

        base = self.initial_base
        if base is None:
            return None

        # If route105 is already the native family, use route123 for the full
        # structural window. If stage-1 route105 is injected by the exact gate,
        # let it run 144..191, then use route123 192..239.
        if base == 105 and 144 <= step < 240:
            return 123
        if base in COW_BRANCH_SOURCE_ROUTES and 192 <= step < 240:
            return 123
        return None


def run_game(task):
    mode, seed, seat, snapshot = task
    runtime = ContinuousRuntime(
        ROOT / "assets" / "parent_promoted_v2.pt",
        snapshot,
        stochastic=False,
        temperature=0.90,
        residual_scale=1.0,
        base_keep_bias=2.3,
        decision_every=24,
        enable_structural_route_gate=True,
        seed=int(seed) * 17 + int(seat),
    )
    structural_policy = (
        TwoStageStructuralPolicy() if mode == "two_stage" else None
    )
    learner = V4HybridRolloutAgent(
        option_policy=runtime,
        structural_route_policy=structural_policy,
        min_option_confidence=0.0,
        enable_market_race_ordering=True,
        enable_idle_weed_labor=True,
        enable_structural_route_gate=True,
    )
    opponent = str(ROOT / "assets" / "v51_main.py")
    agents = [learner, opponent] if seat == 0 else [opponent, learner]
    env = make(
        "kaggriculture",
        configuration={"seed": int(seed), "episodeSteps": 720},
        debug=False,
    )
    env.run(agents)
    farms = env.steps[-1][0].observation.farms
    own = int(farms[seat].money)
    rival = int(farms[1 - seat].money)
    return {
        "mode": mode,
        "seed": int(seed),
        "seat": int(seat),
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
    known = [(int(g["seed"]), int(g["seat"])) for g in winners]
    heldout = [
        (seed, seat)
        for seed in range(22990000, 22990004)
        for seat in (0, 1)
    ]
    all_games = known + heldout
    snapshot = ROOT / "output" / "causal_v4_1_actor.pt"

    result = {}
    ctx = mp.get_context("spawn")
    for mode in ("gate_only", "two_stage"):
        tasks = [(mode, s, seat, snapshot) for s, seat in all_games]
        with ctx.Pool(processes=6) as pool:
            rows = list(pool.imap_unordered(run_game, tasks, chunksize=1))
        known_rows = [
            r for r in rows if (r["seed"], r["seat"]) in set(known)
        ]
        held_rows = [
            r for r in rows if (r["seed"], r["seat"]) in set(heldout)
        ]
        result[mode] = {
            "known": sorted(known_rows, key=lambda r: (r["seed"], r["seat"])),
            "known_metrics": metrics(known_rows),
            "heldout": sorted(held_rows, key=lambda r: (r["seed"], r["seat"])),
            "heldout_metrics": metrics(held_rows),
        }
        print(
            mode,
            "KNOWN",
            result[mode]["known_metrics"],
            "HELDOUT",
            result[mode]["heldout_metrics"],
            flush=True,
        )

    out = ROOT / "output" / "structural_stage2_ab_v4_2.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    b = result["gate_only"]["heldout_metrics"]["mean_margin"]
    t = result["two_stage"]["heldout_metrics"]["mean_margin"]
    print(f"DELTA_HELDOUT {t-b:+.1f}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
