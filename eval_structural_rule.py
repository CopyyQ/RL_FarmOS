import copy
import json
import multiprocessing as mp
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT), str(ROOT / "src")]

from kaggle_environments import make
from continuous_runtime import ContinuousRuntime, ScriptedTrajectoryRuntime, HORIZONS
from rollout.v4_hybrid_agent import V4HybridRolloutAgent


def final_margin(env, seat):
    farms = env.steps[-1][0].observation.farms
    own = int(farms[seat].money)
    rival = int(farms[1 - seat].money)
    return own - rival, own, rival


def run_actor(seed, seat):
    rt = ContinuousRuntime(
        ROOT / "assets" / "parent_promoted_v2.pt",
        ROOT / "output" / "causal_v4_1_actor.pt",
        stochastic=False,
        temperature=0.90,
        residual_scale=1.0,
        base_keep_bias=2.3,
        decision_every=24,
        enable_structural_route_gate=False,
        seed=int(seed) * 13 + int(seat),
    )
    learner = V4HybridRolloutAgent(
        option_policy=rt,
        min_option_confidence=0.0,
        enable_market_race_ordering=True,
        enable_idle_weed_labor=True,
        enable_structural_route_gate=False,
    )
    opp = str(ROOT / "assets" / "v51_main.py")
    agents = [learner, opp] if seat == 0 else [opp, learner]
    env = make(
        "kaggriculture",
        configuration={"seed": int(seed), "episodeSteps": 720},
        debug=False,
    )
    env.run(agents)
    margin, own, rival = final_margin(env, seat)
    return {
        "margin": margin,
        "own": own,
        "v51": rival,
        "records": copy.deepcopy(rt.records),
    }


def mutate(records, route_to_class, horizon):
    rows = copy.deepcopy(records)
    target = None
    for r in rows:
        if int(r["step"]) == 144:
            target = r
            break
    if target is None:
        return None
    target["chosen_route_id"] = 105
    target["route_action"] = int(route_to_class[105])
    target["horizon"] = int(horizon)
    target["horizon_action"] = HORIZONS.index(int(horizon))
    return rows


def run_script(seed, seat, records):
    rt = ScriptedTrajectoryRuntime(
        ROOT / "assets" / "parent_promoted_v2.pt",
        records,
    )
    learner = V4HybridRolloutAgent(
        option_policy=rt,
        min_option_confidence=0.0,
        enable_market_race_ordering=True,
        enable_idle_weed_labor=True,
        allow_all_routes=True,
        enable_structural_route_gate=False,
    )
    opp = str(ROOT / "assets" / "v51_main.py")
    agents = [learner, opp] if seat == 0 else [opp, learner]
    env = make(
        "kaggriculture",
        configuration={"seed": int(seed), "episodeSteps": 720},
        debug=False,
    )
    env.run(agents)
    margin, own, rival = final_margin(env, seat)
    return {
        "margin": margin,
        "own": own,
        "v51": rival,
        "mismatches": list(rt.mismatches),
    }


def worker(task):
    seed, seat, route_to_class = task
    actor = run_actor(seed, seat)
    scripted = run_script(seed, seat, actor["records"])
    row = {
        "seed": int(seed),
        "seat": int(seat),
        "actor_margin": int(actor["margin"]),
        "scripted_margin": int(scripted["margin"]),
        "scripted_mismatches": scripted["mismatches"],
    }
    for horizon in (48, 72):
        changed = mutate(actor["records"], route_to_class, horizon)
        if changed is None:
            row[f"route105_h{horizon}_margin"] = None
            row[f"route105_h{horizon}_gain"] = None
            continue
        result = run_script(seed, seat, changed)
        row[f"route105_h{horizon}_margin"] = int(result["margin"])
        row[f"route105_h{horizon}_gain"] = int(result["margin"]) - int(scripted["margin"])
    return row


def main():
    payload = torch.load(
        ROOT / "assets" / "parent_promoted_v2.pt",
        map_location="cpu",
        weights_only=False,
    )
    route_ids = tuple(int(x) for x in payload["route_ids"])
    route_to_class = {rid: i for i, rid in enumerate(route_ids)}
    tasks = [
        (seed, seat, route_to_class)
        for seed in range(22990000, 22990004)
        for seat in (0, 1)
    ]
    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(processes=4) as pool:
        for i, row in enumerate(pool.imap_unordered(worker, tasks, chunksize=1), 1):
            rows.append(row)
            print("[STRUCT-HELDOUT]", i, "/", len(tasks), row, flush=True)
    rows.sort(key=lambda x: (x["seed"], x["seat"]))
    out = {"schema": "farmos_structural_rule_eval_v4_2", "rows": rows}
    for h in (48, 72):
        gains = [r[f"route105_h{h}_gain"] for r in rows if r[f"route105_h{h}_gain"] is not None]
        margins = [r[f"route105_h{h}_margin"] for r in rows if r[f"route105_h{h}_margin"] is not None]
        out[f"h{h}"] = {
            "mean_gain": sum(gains) / len(gains),
            "positive": sum(x > 0 for x in gains),
            "neutral": sum(x == 0 for x in gains),
            "negative": sum(x < 0 for x in gains),
            "mean_margin": sum(margins) / len(margins),
            "wins": sum(x > 0 for x in margins),
        }
    (ROOT / "output" / "structural_rule_heldout_v4_2.json").write_text(
        json.dumps(out, indent=2),
        encoding="utf-8",
    )
    print("SUMMARY", out["h48"], out["h72"], flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
