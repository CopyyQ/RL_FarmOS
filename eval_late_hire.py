import json
import multiprocessing as mp
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/"vendor"), str(ROOT), str(ROOT/"src")]

from kaggle_environments import make
from continuous_runtime import ContinuousRuntime
from rollout.v4_hybrid_agent import V4HybridRolloutAgent


def metrics(rows):
    margins = [int(x["margin"]) for x in rows]
    wins = sum(int(x["win"]) for x in rows)
    return {
        "games": len(rows),
        "wins": wins,
        "win_rate": wins / max(1, len(rows)),
        "own_money": statistics.fmean(x["own_money"] for x in rows),
        "v51_money": statistics.fmean(x["v51_money"] for x in rows),
        "mean_margin": statistics.fmean(margins),
        "median_margin": statistics.median(margins),
        "min_margin": min(margins),
        "max_margin": max(margins),
    }


def run_one(task):
    seed, seat, snapshot, enable_prune, cfg = task
    runtime = ContinuousRuntime(
        ROOT/"assets"/"parent_promoted_v2.pt",
        snapshot,
        stochastic=False,
        temperature=float(cfg.get("min_temperature", 0.9)),
        residual_scale=float(cfg.get("residual_scale", 1.0)),
        base_keep_bias=float(cfg.get("base_keep_bias", 2.3)),
        decision_every=int(cfg.get("decision_every", 24)),
        seed=seed * 31 + seat,
    )
    learner = V4HybridRolloutAgent(
        option_policy=runtime,
        min_option_confidence=0.0,
        enable_market_race_ordering=True,
        enable_idle_weed_labor=True,
        enable_late_hire_pruning=bool(enable_prune),
    )
    opponent = str(ROOT/"assets"/"v51_main.py")
    agents = [learner, opponent] if seat == 0 else [opponent, learner]
    env = make(
        "kaggriculture",
        configuration={"seed": int(seed), "episodeSteps": 720},
        debug=False,
    )
    env.run(agents)
    final = env.steps[-1]
    farms = final[0].observation.farms
    own = int(farms[seat].money)
    rival = int(farms[1-seat].money)
    return {
        "seed": int(seed),
        "seat": int(seat),
        "own_money": own,
        "v51_money": rival,
        "margin": own-rival,
        "win": int(own > rival),
    }


def run_batch(snapshot, games, enabled, cfg, workers):
    tasks = [
        (seed, seat, str(snapshot), enabled, cfg)
        for seed, seat in games
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:
        return list(pool.imap(run_one, tasks, chunksize=1))


def main():
    latest = torch.load(
        ROOT/"output"/"winner_v4_latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    cfg = dict(latest.get("config", {}))
    mem = torch.load(
        ROOT/"output"/"winner_memory_v4.pt",
        map_location="cpu",
        weights_only=False,
    )
    wins = sorted(
        [g for g in mem["games"] if int(g["margin"]) > 0],
        key=lambda g: int(g["margin"]),
        reverse=True,
    )[:3]
    known = [(int(g["seed"]), int(g["seat"])) for g in wins]
    heldout = [
        (seed, seat)
        for seed in range(22990000, 22990004)
        for seat in (0, 1)
    ]
    snapshot = (
        ROOT/"output"/"causal_v4_1_actor.pt"
        if (ROOT/"output"/"causal_v4_1_actor.pt").exists()
        else ROOT/"output"/"winner_v4_latest.pt"
    )
    result = {}
    for enabled in (False, True):
        key = "prune_on" if enabled else "prune_off"
        known_rows = run_batch(snapshot, known, enabled, cfg, 3)
        held_rows = run_batch(snapshot, heldout, enabled, cfg, 4)
        result[key] = {
            "known": known_rows,
            "known_metrics": metrics(known_rows),
            "heldout": held_rows,
            "heldout_metrics": metrics(held_rows),
        }
        print(key, "KNOWN", result[key]["known_metrics"], flush=True)
        print(key, "HELDOUT", result[key]["heldout_metrics"], flush=True)

    off = result["prune_off"]
    on = result["prune_on"]
    print(
        "DELTA known_mean_margin="
        f"{on['known_metrics']['mean_margin']-off['known_metrics']['mean_margin']:+.1f} "
        "heldout_mean_margin="
        f"{on['heldout_metrics']['mean_margin']-off['heldout_metrics']['mean_margin']:+.1f} "
        f"known_wins={off['known_metrics']['wins']}->{on['known_metrics']['wins']}",
        flush=True,
    )
    (ROOT/"output"/"late_hire_ab_v4_2.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
