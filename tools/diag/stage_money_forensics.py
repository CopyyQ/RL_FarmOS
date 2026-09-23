from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src"), str(ROOT)]

from kaggle_environments import make, environments
from kaggle_environments.envs.kaggriculture import kaggriculture as kg

from kaggrl.economic_value import economic_value_snapshot
from rollout.v4_hybrid_agent import V4HybridRolloutAgent
from v45_skill_runtime_actkeep import V45SkillActKeepRuntime
from winner_train import ALLOWED_MARKETS


ITERATION = 1205
TEMPERATURE = 1.00242003614173
PARENT = str(ROOT / "assets" / "parent_promoted_v2.pt")
SNAPSHOT = str(ROOT / "assets" / "tournament_actor_iter1205.pt")
OPPONENT = str(ROOT / "assets" / "v51_main.py")
GATE = str(ROOT / "assets" / "act_keep_gate_sweep_best.pt")
CONFIG = {"episodeSteps": 720, "turnsPerDay": 24}

PHASES = [
    ("A_PRE_SHOP", 0, 2),
    ("B_EARLY_SHOP", 3, 8),
    ("C_SIGNAL_BUILD", 9, 14),
    ("D_PORTFOLIO_DIVERGE", 15, 20),
    ("E_LATENT_GAP", 21, 24),
    ("F_CASH_CONVERSION", 25, 29),
]

KEY_PRODUCTS = ("WHEAT", "TOMATO", "STRAWBERRY", "MILK", "WOOL", "EGG")


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _farm_counts(farm):
    crops = Counter()
    animals = Counter()
    weeds = 0
    structures = Counter()
    for row in list(_get(farm, "tiles", []) or []):
        for tile in list(row or []):
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                crops[str(tile.get("crop", ""))] += 1
            if tile.get("animal"):
                animals[str(tile["animal"])] += 1
            if tile.get("kind") == "WEED":
                weeds += 1
            kind = str(tile.get("kind", ""))
            if kind in {"COOP", "PASTURE"}:
                structures[kind] += 1
    return {
        "crops": dict(crops),
        "animals": dict(animals),
        "weeds": weeds,
        "structures": dict(structures),
        "hands": len(list(_get(farm, "hands", []) or [])),
        "land": len(list(_get(farm, "unlocked_quadrants", []) or [])),
    }


def _private_holdings(obs):
    private = _get(obs, "private", {}) or {}
    shed = dict(_get(private, "shed", {}) or {})
    seeds = dict(_get(private, "seeds", {}) or {})
    return {
        "shed": {k: int(v or 0) for k, v in shed.items() if int(v or 0)},
        "seeds": {k: int(v or 0) for k, v in seeds.items() if int(v or 0)},
    }


class MarketRecorder:
    def __init__(self):
        self.events = []
        self.step = -1
        self.farms = []
        self._orig_process = kg._process_market
        self._orig_commit = kg._commit_unit
        self._orig_hire = kg._do_hire
        self._orig_land = kg._do_buy_land
        self._orig_registered = environments["kaggriculture"]["interpreter"]

    def player_id(self, farm):
        for idx, candidate in enumerate(self.farms):
            if candidate is farm:
                return idx
        return -1

    def install(self):
        recorder = self

        def process(state, env):
            obs0 = state[0].observation
            recorder.step = int(_get(obs0, "step", 0) or 0)
            recorder.farms = list(_get(obs0, "farms", []) or [])
            return recorder._orig_process(state, env)

        def commit(op, item, price, farm, private, market, shed_capacity=100):
            before = int(_get(farm, "money", 0) or 0)
            ok = recorder._orig_commit(
                op, item, price, farm, private, market, shed_capacity
            )
            after = int(_get(farm, "money", 0) or 0)
            if ok:
                recorder.events.append(
                    {
                        "step": recorder.step,
                        "day": recorder.step // 24,
                        "player": recorder.player_id(farm),
                        "kind": str(op),
                        "item": str(item),
                        "money_delta": after - before,
                        "price": int(price),
                        "units": 1,
                    }
                )
            return ok

        def hire(farm, private, board_size, mult=kg.FARM_HAND_COST_MULT):
            before = int(_get(farm, "money", 0) or 0)
            recorder._orig_hire(farm, private, board_size, mult)
            after = int(_get(farm, "money", 0) or 0)
            if after != before:
                recorder.events.append(
                    {
                        "step": recorder.step,
                        "day": recorder.step // 24,
                        "player": recorder.player_id(farm),
                        "kind": "HIRE",
                        "item": "",
                        "money_delta": after - before,
                        "price": before - after,
                        "units": 1,
                    }
                )

        def land(farm, board_size):
            before = int(_get(farm, "money", 0) or 0)
            recorder._orig_land(farm, board_size)
            after = int(_get(farm, "money", 0) or 0)
            if after != before:
                recorder.events.append(
                    {
                        "step": recorder.step,
                        "day": recorder.step // 24,
                        "player": recorder.player_id(farm),
                        "kind": "BUY_LAND",
                        "item": "",
                        "money_delta": after - before,
                        "price": before - after,
                        "units": 1,
                    }
                )

        kg._process_market = process
        kg._commit_unit = commit
        kg._do_hire = hire
        kg._do_buy_land = land

    def restore(self):
        kg._process_market = self._orig_process
        kg._commit_unit = self._orig_commit
        kg._do_hire = self._orig_hire
        kg._do_buy_land = self._orig_land


def make_agent(seed, seat):
    runtime = V45SkillActKeepRuntime(
        PARENT,
        SNAPSHOT,
        stochastic=True,
        temperature=TEMPERATURE,
        residual_scale=1.0,
        base_keep_bias=2.3,
        decision_every=24,
        allowed_market_modes=ALLOWED_MARKETS,
        skill_cutover_step=672,
        skill_keep_penalty=2.75,
        skill_shadow_start_step=0,
        skill_confidence_threshold=0.70,
        act_keep_gate_path=GATE,
        act_keep_threshold=0.50,
        seed=ITERATION * 1_000_003 + 17,
    )
    runtime.reset()
    runtime.rng.manual_seed(
        int(seed) * 1009 + int(seat) * 97 + ITERATION * 1_000_003
    )
    learner = V4HybridRolloutAgent(
        option_policy=runtime,
        min_option_confidence=0.0,
        enable_market_race_ordering=True,
    )
    learner.reset()
    return learner


def phase_events(events, player, d0, d1):
    rows = [
        e for e in events
        if e["player"] == player and d0 <= e["day"] <= d1
    ]
    sales = defaultdict(lambda: {"units": 0, "revenue": 0})
    spend = defaultdict(lambda: {"units": 0, "cost": 0})
    for row in rows:
        kind = row["kind"]
        item = row["item"]
        if kind == "SELL":
            sales[item]["units"] += 1
            sales[item]["revenue"] += int(row["money_delta"])
        elif row["money_delta"] < 0:
            key = f"{kind}:{item}" if item else kind
            spend[key]["units"] += 1
            spend[key]["cost"] += -int(row["money_delta"])
    total_sales = sum(v["revenue"] for v in sales.values())
    total_spend = sum(v["cost"] for v in spend.values())
    return {
        "sales": dict(sales),
        "spend": dict(spend),
        "sales_total": total_sales,
        "spend_total": total_spend,
        "net_market_cash": total_sales - total_spend,
    }


def snapshot(env, idx, seat):
    idx = max(0, min(int(idx), len(env.steps) - 1))
    state = env.steps[idx][seat]
    obs = state.observation
    farm = obs.farms[seat]
    econ = economic_value_snapshot(obs, CONFIG, horizon_steps=72)
    market = _get(obs, "market", {}) or {}
    prices = dict(_get(market, "prices", {}) or {})
    town = _get(obs, "town", {}) or {}
    return {
        "idx": idx,
        "day": int(_get(obs, "day", idx // 24) or 0),
        "money": int(_get(farm, "money", 0) or 0),
        "farm": _farm_counts(farm),
        "private": _private_holdings(obs),
        "current_net_worth": int(econ["own_current_net_worth"]),
        "horizon_value": int(econ["own_horizon_value"]),
        "horizon_output": {
            k: round(float(econ["own_visible_output"].get(k, 0.0)), 2)
            for k in KEY_PRODUCTS
        },
        "prices": {k: int(prices.get(k, 0) or 0) for k in KEY_PRODUCTS},
        "shops": list(_get(town, "unlocked_shops", []) or []),
    }


def run_game(seed, our_seat):
    recorder = MarketRecorder()
    recorder.install()
    try:
        learner = make_agent(seed, our_seat)
        env = make(
            "kaggriculture",
            configuration={"seed": int(seed), "episodeSteps": 720},
            debug=False,
        )
        env.run(
            [learner, OPPONENT]
            if our_seat == 0
            else [OPPONENT, learner]
        )
    finally:
        recorder.restore()

    final_farms = env.steps[-1][0].observation.farms
    rival_seat = 1 - our_seat
    result = {
        "seed": int(seed),
        "our_seat": int(our_seat),
        "final_margin": int(
            final_farms[our_seat].money - final_farms[rival_seat].money
        ),
        "phases": [],
    }

    for name, d0, d1 in PHASES:
        start_idx = d0 * 24
        end_idx = min(719, (d1 + 1) * 24)
        ours0 = snapshot(env, start_idx, our_seat)
        ours1 = snapshot(env, end_idx, our_seat)
        opp0 = snapshot(env, start_idx, rival_seat)
        opp1 = snapshot(env, end_idx, rival_seat)
        own_tx = phase_events(recorder.events, our_seat, d0, d1)
        opp_tx = phase_events(recorder.events, rival_seat, d0, d1)

        cash_gap0 = ours0["money"] - opp0["money"]
        cash_gap1 = ours1["money"] - opp1["money"]
        horizon_gap0 = ours0["horizon_value"] - opp0["horizon_value"]
        horizon_gap1 = ours1["horizon_value"] - opp1["horizon_value"]

        result["phases"].append(
            {
                "phase": name,
                "days": [d0, d1],
                "cash": {
                    "ours_start": ours0["money"],
                    "ours_end": ours1["money"],
                    "opp_start": opp0["money"],
                    "opp_end": opp1["money"],
                    "gap_start": cash_gap0,
                    "gap_end": cash_gap1,
                    "gap_change": cash_gap1 - cash_gap0,
                },
                "market_cash": {
                    "ours": own_tx,
                    "opp": opp_tx,
                    "net_advantage": (
                        own_tx["net_market_cash"] - opp_tx["net_market_cash"]
                    ),
                },
                "economic": {
                    "horizon_gap_start": horizon_gap0,
                    "horizon_gap_end": horizon_gap1,
                    "horizon_gap_change": horizon_gap1 - horizon_gap0,
                    "latent_gap_end": horizon_gap1 - cash_gap1,
                },
                "start": {"ours": ours0, "opp": opp0},
                "end": {"ours": ours1, "opp": opp1},
            }
        )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="23238530,23238531")
    ap.add_argument("--both-seats", action="store_true")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    outputs = []
    for seed_text in args.seeds.split(","):
        seed = int(seed_text.strip())
        seats = (0, 1) if args.both_seats else ((seed + ITERATION) & 1,)
        for seat in seats:
            outputs.append(run_game(seed, int(seat)))

    payload = {"games": outputs}
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
