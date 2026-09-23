import argparse
import copy
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT), str(ROOT / "src")]

from kaggle_environments import make
from kaggrl.clock import resolve_clock
from kaggrl.constants import CROPS, ANIMALS, PRODUCTS
from continuous_runtime import ContinuousRuntime, ScriptedTrajectoryRuntime, HORIZONS
from rollout.v4_hybrid_agent import V4HybridRolloutAgent


def get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return getattr(obj, key)
    except Exception:
        return default


def farm_features(farm, prefix):
    out = {}
    out[f"{prefix}_money"] = float(get(farm, "money", 0) or 0)
    out[f"{prefix}_hands"] = float(len(get(farm, "hands", []) or []))
    out[f"{prefix}_hires"] = float(get(farm, "hires_today", 0) or 0)
    counts = {"EMPTY": 0, "WEED": 0, "PLANT": 0, "COOP": 0, "PASTURE": 0}
    crops = {x: 0 for x in CROPS}
    animals = {x: 0 for x in ANIMALS}
    yields = {x: 0.0 for x in CROPS}
    for row in get(farm, "tiles", []) or []:
        for tile in row:
            if tile is None:
                counts["EMPTY"] += 1
                continue
            if tile == "LOCKED":
                continue
            kind = str(get(tile, "kind", "") or "")
            if kind in counts:
                counts[kind] += 1
            crop = get(tile, "crop")
            if crop in crops:
                crops[crop] += 1
                yields[crop] += float(get(tile, "yield_units", 0) or 0)
            animal = get(tile, "animal")
            if animal in animals:
                animals[animal] += 1
    for k, v in counts.items():
        out[f"{prefix}_{k.lower()}"] = float(v)
    for k, v in crops.items():
        out[f"{prefix}_crop_{k.lower()}"] = float(v)
        out[f"{prefix}_yield_{k.lower()}"] = float(yields[k])
    for k, v in animals.items():
        out[f"{prefix}_animal_{k.lower()}"] = float(v)
    return out


def economic_features(observation):
    player = int(get(observation, "player", 0) or 0)
    farms = list(get(observation, "farms", []) or [])
    own = farms[player]
    rival = farms[1 - player]
    out = {}
    out.update(farm_features(own, "own"))
    out.update(farm_features(rival, "rival"))
    out["money_diff"] = out["own_money"] - out["rival_money"]
    market = get(observation, "market", {}) or {}
    inv = get(market, "inventory", {}) or {}
    prices = get(market, "prices", {}) or {}
    for item in PRODUCTS:
        out[f"market_inv_{item.lower()}"] = float(get(inv, item, 0) or 0)
        out[f"market_price_{item.lower()}"] = float(get(prices, item, 0) or 0)
    private = get(observation, "private", {}) or {}
    shed = get(private, "shed", {}) or {}
    seeds = get(private, "seeds", {}) or {}
    for item in PRODUCTS:
        out[f"shed_{item.lower()}"] = float(get(shed, item, 0) or 0)
    for crop in CROPS:
        out[f"seed_{crop.lower()}"] = float(get(seeds, crop, 0) or 0)
    town = get(observation, "town", {}) or {}
    out["town_unlocked_shops"] = float(len(get(town, "unlocked_shops", []) or []))
    return out


class CaptureRuntime(ContinuousRuntime):
    def reset(self):
        super().reset()
        self.capture144 = None

    def __call__(self, observation, configuration, context):
        step = int(resolve_clock(observation, configuration).step)
        if step == 144 and self.capture144 is None:
            self.capture144 = economic_features(observation)
        return super().__call__(observation, configuration, context)
def final_margin(env, seat):
    farms = env.steps[-1][0].observation.farms
    own = int(farms[seat].money)
    rival = int(farms[1 - seat].money)
    return own - rival, own, rival


def run_actor(seed, seat):
    rt = CaptureRuntime(
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
    env = make("kaggriculture", configuration={"seed": int(seed), "episodeSteps": 720}, debug=False)
    env.run(agents)
    margin, own, rival = final_margin(env, seat)
    record = next((r for r in rt.records if int(r["step"]) == 144), None)
    if record is None:
        raise RuntimeError(f"seed={seed} seat={seat}: step144 record missing")
    hidden = np.frombuffer(record["hidden_f16"], np.float16).astype(np.float32)
    clock = np.frombuffer(record["clock_f16"], np.float16).astype(np.float32)
    return {
        "margin": margin,
        "own": own,
        "v51": rival,
        "records": copy.deepcopy(rt.records),
        "features": dict(rt.capture144 or {}),
        "hidden": hidden,
        "clock": clock,
        "base_route_id": int(record["base_route_id"]),
        "chosen_route_id": int(record["chosen_route_id"]),
    }


def mutate_route105(records, route_to_class, horizon=48):
    rows = copy.deepcopy(records)
    target = next((r for r in rows if int(r["step"]) == 144), None)
    if target is None:
        raise RuntimeError("step144 record missing")
    target["chosen_route_id"] = 105
    target["route_action"] = int(route_to_class[105])
    target["horizon"] = int(horizon)
    target["horizon_action"] = HORIZONS.index(int(horizon))
    return rows


def run_candidate(seed, seat, records):
    rt = ScriptedTrajectoryRuntime(ROOT / "assets" / "parent_promoted_v2.pt", records)
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
    env = make("kaggriculture", configuration={"seed": int(seed), "episodeSteps": 720}, debug=False)
    env.run(agents)
    margin, own, rival = final_margin(env, seat)
    return margin, own, rival, list(rt.mismatches)


def worker(task):
    seed, seat, route_to_class = task
    base = run_actor(seed, seat)
    changed = mutate_route105(base["records"], route_to_class, 48)
    margin, own, rival, mismatches = run_candidate(seed, seat, changed)
    return {
        "seed": int(seed),
        "seat": int(seat),
        "baseline_margin": int(base["margin"]),
        "candidate_margin": int(margin),
        "gain": int(margin) - int(base["margin"]),
        "base_route_id": int(base["base_route_id"]),
        "chosen_route_id": int(base["chosen_route_id"]),
        "features": base["features"],
        "hidden": base["hidden"],
        "clock": base["clock"],
        "mismatches": mismatches,
    }
def save(rows, output):
    payload = {
        "schema": "farmos_route105_structural_gate_dataset_v4_2",
        "candidate_route": 105,
        "horizon": 48,
        "rows": rows,
    }
    torch.save(payload, output)
    compact = []
    for r in rows:
        compact.append({
            "seed": r["seed"],
            "seat": r["seat"],
            "baseline_margin": r["baseline_margin"],
            "candidate_margin": r["candidate_margin"],
            "gain": r["gain"],
            "base_route_id": r["base_route_id"],
            "chosen_route_id": r["chosen_route_id"],
            "features": r["features"],
            "mismatches": r["mismatches"],
        })
    Path(output).with_suffix(".json").write_text(json.dumps({
        "schema": payload["schema"],
        "candidate_route": 105,
        "horizon": 48,
        "rows": compact,
    }, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=22991000)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", default=str(ROOT / "output" / "route105_gate_dataset_v4_2.pt"))
    args = ap.parse_args()
    payload = torch.load(ROOT / "assets" / "parent_promoted_v2.pt", map_location="cpu", weights_only=False)
    route_ids = tuple(int(x) for x in payload["route_ids"])
    route_to_class = {rid: i for i, rid in enumerate(route_ids)}
    tasks = [(args.seed_start + i, args.seat, route_to_class) for i in range(args.seeds)]
    rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(worker, tasks, chunksize=1), 1):
            rows.append(row)
            rows.sort(key=lambda x: (x["seed"], x["seat"]))
            save(rows, args.output)
            print(
                f"[ROUTE105-DATA] {i}/{len(tasks)} seed={row['seed']} "
                f"base={row['base_route_id']} gain={row['gain']:+d} "
                f"margin={row['candidate_margin']:+d}",
                flush=True,
            )
    gains = [int(r["gain"]) for r in rows]
    print(
        f"DONE n={len(rows)} mean_gain={sum(gains)/len(gains):+.1f} "
        f"positive={sum(x>0 for x in gains)} neutral={sum(x==0 for x in gains)} "
        f"negative={sum(x<0 for x in gains)}",
        flush=True,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
