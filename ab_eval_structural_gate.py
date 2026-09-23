import json
import multiprocessing as mp
import sys
from pathlib import Path

import torch

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
    return {
        "seed": int(seed),
        "seat": int(seat),
        "gate": bool(gate),
        "margin": own - rival,
        "own": own,
        "v51": rival,
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
    mem = torch.load(
        ROOT / "output" / "winner_memory_v4.pt",
        map_location="cpu",
        weights_only=False,
    )
    wins = sorted(
        [g for g in mem["games"] if int(g["margin"]) > 0],
        key=lambda g: int(g["margin"]),
        reverse=True,
    )[:3]
    known_games = [(int(g["seed"]), int(g["seat"])) for g in wins]
    heldout_games = [
        (seed, seat)
        for seed in range(22990000, 22990004)
        for seat in (0, 1)
    ]
    games = known_games + heldout_games

    out = {}
    ctx = mp.get_context("spawn")
    for name, gate in (("baseline", False), ("structural_gate", True)):
        tasks = [(seed, seat, gate) for seed, seat in games]
        rows = []
        with ctx.Pool(processes=6) as pool:
            for i, row in enumerate(
                pool.imap_unordered(run, tasks, chunksize=1), 1
            ):
                rows.append(row)
                print(f"[{name}] {i}/{len(tasks)} {row}", flush=True)
        known = [
            x for x in rows
            if (x["seed"], x["seat"]) in set(known_games)
        ]
        heldout = [
            x for x in rows
            if (x["seed"], x["seat"]) not in set(known_games)
        ]
        out[name] = {
            "known": sorted(known, key=lambda x: (x["seed"], x["seat"])),
            "known_metrics": metrics(known),
            "heldout": sorted(heldout, key=lambda x: (x["seed"], x["seat"])),
            "heldout_metrics": metrics(heldout),
        }
        print(
            name,
            "KNOWN", out[name]["known_metrics"],
            "HELDOUT", out[name]["heldout_metrics"],
            flush=True,
        )

    b = out["baseline"]["heldout_metrics"]["mean_margin"]
    g = out["structural_gate"]["heldout_metrics"]["mean_margin"]
    out["delta"] = {
        "heldout_mean_margin": g - b,
        "known_wins_before": out["baseline"]["known_metrics"]["wins"],
        "known_wins_after": out["structural_gate"]["known_metrics"]["wins"],
    }
    (ROOT / "output" / "structural_gate_ab_v4_2.json").write_text(
        json.dumps(out, indent=2),
        encoding="utf-8",
    )
    print("DELTA", out["delta"], flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
