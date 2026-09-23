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


def tile_profile(observation):
    player = int(observation.get("player", 0))
    farm = observation["farms"][player]
    counts = {}
    for row in farm.get("tiles") or []:
        for tile in row:
            if tile is None:
                key = "EMPTY"
            elif tile == "LOCKED":
                key = "LOCKED"
            elif isinstance(tile, dict) and "animal" in tile:
                key = f"ANIMAL_{tile['animal']}"
            elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                key = f"PLANT_{tile.get('crop')}"
            elif isinstance(tile, dict):
                key = str(tile.get("kind", "OTHER"))
            else:
                key = "OTHER"
            counts[key] = counts.get(key, 0) + 1
    return counts


class Recorder:
    def __init__(self):
        self.snapshot = None

    def __call__(self, observation, configuration, selected_route, context):
        if self.snapshot is not None:
            return None
        step = int(resolve_clock(observation, configuration).step)
        if step < 144:
            return None
        player = int(observation.get("player", 0))
        farm = observation["farms"][player]
        market = observation.get("market") or {}
        prices = market.get("prices") or {}
        inv = market.get("inventory") or {}
        self.snapshot = {
            "step": step,
            "shops": list(
                (observation.get("town") or {}).get("unlocked_shops") or []
            ),
            "base_route": int(context.base_route_id),
            "actor_route": int(selected_route),
            "money": int(farm.get("money", 0)),
            "hires_today": int(farm.get("hires_today", 0)),
            "hands": len(farm.get("hands") or []),
            "prices": {
                item: int(prices.get(item, 0))
                for item in ("MILK", "WOOL", "EGG", "WHEAT", "TOMATO", "STRAWBERRY")
            },
            "inventory": {
                item: int(inv.get(item, 0))
                for item in ("MILK", "WOOL", "EGG", "WHEAT", "TOMATO", "STRAWBERRY")
            },
            "tiles": tile_profile(observation),
            "shed": dict((observation.get("private") or {}).get("shed") or {}),
        }
        return None


def run(task):
    seed, seat, group = task
    recorder = Recorder()
    rt = ContinuousRuntime(
        ROOT / "assets" / "parent_promoted_v2.pt",
        ROOT / "output" / "causal_v4_1_actor.pt",
        stochastic=False,
        temperature=0.90,
        residual_scale=1.0,
        base_keep_bias=2.3,
        decision_every=24,
        seed=int(seed) * 43 + int(seat),
    )
    learner = V4HybridRolloutAgent(
        option_policy=rt,
        structural_route_policy=recorder,
        min_option_confidence=0.0,
        enable_market_race_ordering=True,
        enable_idle_weed_labor=True,
        enable_late_hire_pruning=False,
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
    rival = int(farms[1-seat].money)
    return {
        "seed": int(seed),
        "seat": int(seat),
        "group": group,
        "final_margin": own - rival,
        "snapshot": recorder.snapshot,
    }


def load_deltas():
    result = {}
    dev = json.loads(
        (ROOT / "output" / "structural_overlay_ab_v4_2.json").read_text()
    )
    fresh = json.loads(
        (ROOT / "output" / "structural_overlay_fresh16_v4_2.json").read_text()
    )
    for label, data in (("dev", dev), ("fresh", fresh)):
        base_rows = data["baseline"].get("heldout", data["baseline"].get("rows", []))
        base = {(r["seed"], r["seat"]): r for r in base_rows}
        for mode in ("cow105_144_216", "cow123_144_240"):
            rows = data[mode].get("heldout", data[mode].get("rows", []))
            for row in rows:
                key = (int(row["seed"]), int(row["seat"]))
                result[(label, key[0], key[1], mode)] = (
                    int(row["margin"]) - int(base[key]["margin"])
                )
    return result


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
    tasks = [
        (int(g["seed"]), int(g["seat"]), "winner")
        for g in winners
    ]
    tasks += [(s, 0, "dev") for s in range(22990000, 22990004)]
    tasks += [(s, 0, "fresh") for s in range(22990100, 22990108)]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=4) as pool:
        rows = list(pool.imap_unordered(run, tasks, chunksize=1))
    rows.sort(key=lambda r: (r["group"], r["seed"], r["seat"]))

    deltas = load_deltas()
    for row in rows:
        group = row["group"]
        seed = row["seed"]
        seat = row["seat"]
        if group in ("dev", "fresh"):
            row["delta105"] = deltas.get(
                (group, seed, seat, "cow105_144_216")
            )
            row["delta123"] = deltas.get(
                (group, seed, seat, "cow123_144_240")
            )

    out = ROOT / "output" / "structural_gate_diagnostics_v4_2.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows:
        s = row["snapshot"] or {}
        print(
            row["group"], row["seed"], row["seat"],
            "shops=", s.get("shops"),
            "base=", s.get("base_route"),
            "actor=", s.get("actor_route"),
            "milk=", (s.get("prices") or {}).get("MILK"),
            "wool=", (s.get("prices") or {}).get("WOOL"),
            "money=", s.get("money"),
            "d105=", row.get("delta105"),
            "d123=", row.get("delta123"),
            flush=True,
        )
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
