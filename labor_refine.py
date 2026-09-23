import argparse
import copy
import json
import multiprocessing as mp
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/"vendor"), str(ROOT), str(ROOT/"src")]

from kaggle_environments import make
from kaggrl.clock import resolve_clock
from continuous_runtime import ScriptedTrajectoryRuntime
from rollout.v4_hybrid_agent import V4HybridRolloutAgent


def fib(n):
    a, b = 1, 1
    for _ in range(max(0, int(n))):
        a, b = b, a+b
    return a


def load_winners(path, limit):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    games = [g for g in payload.get("games", []) if int(g.get("margin", 0)) > 0]
    games.sort(key=lambda g: int(g["margin"]), reverse=True)
    return games[:max(1, int(limit))]


class HireProbeAgent:
    def __init__(self, game, parent, opponent, target_step=None):
        self.game = game
        self.runtime = ScriptedTrajectoryRuntime(parent, game["records"])
        self.agent = V4HybridRolloutAgent(
            option_policy=self.runtime,
            min_option_confidence=0.0,
            enable_market_race_ordering=True,
            enable_idle_weed_labor=True,
        )
        self.target_step = (
            None if target_step is None else int(target_step)
        )
        self.events = []
        self.removed = False

    def __call__(self, observation, configuration=None):
        action = self.agent(observation, configuration)
        clock = resolve_clock(observation, configuration)
        step = int(clock.step)
        player = int(observation.get("player", 0))
        farm = observation["farms"][player]
        market = [list(x) for x in (action.get("market") or [])]
        hire_idx = [
            i for i, x in enumerate(market)
            if x and str(x[0]) == "HIRE"
        ]
        if hire_idx:
            n = int(farm.get("hires_today", 0))
            self.events.append(
                {
                    "step": step,
                    "day": int(clock.day),
                    "hour": int(clock.hour),
                    "hires_today_before": n,
                    "marginal_cost": fib(n),
                    "hire_orders": len(hire_idx),
                    "hands_before": len(farm.get("hands") or []),
                    "money_before": int(farm.get("money", 0)),
                }
            )
        if (
            self.target_step is not None
            and step == self.target_step
            and hire_idx
            and not self.removed
        ):
            del market[hire_idx[-1]]
            action = copy.deepcopy(action)
            action["market"] = market
            self.removed = True
        return action


def run_game(game, parent, opponent, target_step=None):
    learner = HireProbeAgent(
        game, parent, opponent, target_step=target_step
    )
    seat = int(game["seat"])
    seed = int(game["seed"])
    opp = str(opponent)
    agents = [learner, opp] if seat == 0 else [opp, learner]
    env = make(
        "kaggriculture",
        configuration={"seed": seed, "episodeSteps": 720},
        debug=False,
    )
    env.run(agents)
    final = env.steps[-1]
    farms = final[0].observation.farms
    own = int(farms[seat].money)
    rival = int(farms[1-seat].money)
    return {
        "seed": seed,
        "seat": seat,
        "own_money": own,
        "v51_money": rival,
        "margin": own-rival,
        "win": int(own > rival),
        "events": learner.events,
        "removed": learner.removed,
    }


def worker(task):
    game, parent, opponent, step = task
    row = run_game(game, parent, opponent, target_step=step)
    row["target_step"] = int(step)
    return row
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--memory",
        default=str(ROOT/"output"/"winner_memory_v4.pt"),
    )
    ap.add_argument(
        "--parent",
        default=str(ROOT/"assets"/"parent_promoted_v2.pt"),
    )
    ap.add_argument(
        "--opponent",
        default=str(ROOT/"assets"/"v51_main.py"),
    )
    ap.add_argument("--top-winners", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-hire-cost", type=int, default=8)
    ap.add_argument("--max-candidates-per-winner", type=int, default=32)
    ap.add_argument(
        "--output",
        default=str(ROOT/"output"/"labor_refine_v4_2.json"),
    )
    args = ap.parse_args()

    winners = load_winners(Path(args.memory), args.top_winners)
    baselines = {}
    candidates = []
    for game in winners:
        baseline = run_game(
            game, Path(args.parent), Path(args.opponent), None
        )
        key = (int(game["seed"]), int(game["seat"]))
        baselines[key] = baseline
        expensive = [
            e for e in baseline["events"]
            if int(e["marginal_cost"]) >= int(args.min_hire_cost)
        ]
        expensive.sort(
            key=lambda e: (
                -int(e["marginal_cost"]),
                int(e["step"]),
            )
        )
        expensive = expensive[: int(args.max_candidates_per_winner)]
        print(
            f"[LABOR-BASE] seed={key[0]} seat={key[1]} "
            f"margin={baseline['margin']:+d} "
            f"hire_events={len(baseline['events'])} "
            f"expensive={len(expensive)}",
            flush=True,
        )
        for event in expensive[:20]:
            print(
                f"  step={event['step']} day={event['day']} "
                f"hire#{event['hires_today_before']+1} "
                f"cost={event['marginal_cost']} "
                f"hands={event['hands_before']} "
                f"money={event['money_before']}",
                flush=True,
            )
            candidates.append(
                (
                    game,
                    Path(args.parent),
                    Path(args.opponent),
                    int(event["step"]),
                )
            )

    print(f"LABOR tasks={len(candidates)}", flush=True)
    rows = []
    if candidates:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=max(1, int(args.workers)),
            maxtasksperchild=1,
        ) as pool:
            for i, row in enumerate(
                pool.imap_unordered(worker, candidates, chunksize=1),
                1,
            ):
                rows.append(row)
                if i % max(4, int(args.workers)) == 0 or i == len(candidates):
                    print(f"[LABOR] {i}/{len(candidates)}", flush=True)

    for row in rows:
        key = (int(row["seed"]), int(row["seat"]))
        base = baselines[key]
        row["baseline_margin"] = int(base["margin"])
        row["gain_vs_baseline"] = (
            int(row["margin"]) - int(base["margin"])
        )

    improved = [
        r for r in rows
        if bool(r.get("removed"))
        and int(r["gain_vs_baseline"]) > 0
    ]
    improved.sort(
        key=lambda r: int(r["gain_vs_baseline"]),
        reverse=True,
    )
    payload = {
        "schema": "farmos_v51_hire_counterfactual_v4_2",
        "baselines": {
            f"{k[0]}:{k[1]}": v for k, v in baselines.items()
        },
        "task_count": len(candidates),
        "improvement_count": len(improved),
        "rows": rows,
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    gains = [int(r["gain_vs_baseline"]) for r in improved]
    print(
        f"LABOR_DONE improved={len(improved)}/{len(rows)} "
        f"median_positive={statistics.median(gains) if gains else 0:+.1f} "
        f"max_gain={max(gains) if gains else 0:+d}",
        flush=True,
    )
    for row in improved[:30]:
        print(
            f"[LABOR-GAIN] step={row['target_step']} "
            f"gain={row['gain_vs_baseline']:+d} "
            f"margin={row['margin']:+d}",
            flush=True,
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()
