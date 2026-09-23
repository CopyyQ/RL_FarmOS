import argparse
import copy
import csv
import json
import multiprocessing as mp
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kaggle_environments import make
from continuous_runtime import ScriptedTrajectoryRuntime, HORIZONS
from rollout.v4_hybrid_agent import V4HybridRolloutAgent

MARKETS = (
    "KEEP_ROUTE",
    "LIQUIDATE_SHED",
    "HOLD_SALES",
    "FRONT_RUN_1",
    "FRONT_RUN_9",
)
def load_winners(memory_path: Path, limit: int, offset: int = 0):
    payload = torch.load(memory_path, map_location="cpu", weights_only=False)
    games = [g for g in payload.get("games", []) if int(g.get("margin", 0)) > 0]
    games.sort(
        key=lambda g: (int(g["margin"]), int(g["own_money"])),
        reverse=True,
    )
    start = max(0, int(offset))
    end = start + max(1, int(limit))
    return games[start:end]


def mutate_records(game, kind, step, choice):
    records = copy.deepcopy(game["records"])
    found = False
    for record in records:
        if int(record["step"]) != int(step):
            continue
        found = True
        if kind == "market":
            record["market_mode"] = str(choice)
        elif kind == "horizon":
            record["horizon"] = int(choice)
            record["horizon_action"] = HORIZONS.index(int(choice))
        else:
            raise ValueError(kind)
        break
    if not found:
        raise KeyError(f"step {step} not found")
    return records
def probe_market_relevant(game, parent_path, opponent_path):
    runtime = ScriptedTrajectoryRuntime(parent_path, game["records"])
    learner = V4HybridRolloutAgent(
        option_policy=runtime,
        min_option_confidence=0.0,
        enable_market_race_ordering=True,
        enable_structural_route_gate=False,
    )
    seat = int(game["seat"])
    seed = int(game["seed"])
    opponent = str(opponent_path)
    agents = [learner, opponent] if seat == 0 else [opponent, learner]
    env = make(
        "kaggriculture",
        configuration={"seed": seed, "episodeSteps": 720},
        debug=False,
    )
    env.run(agents)
    if len(env.steps) != 720:
        raise RuntimeError(
            f"probe seed={seed} seat={seat}: {len(env.steps)} frames"
        )
    relevant = {
        int(row["step"])
        for row in runtime.applied
        if bool(row.get("market_relevant", False))
    }
    return {
        "seed": seed,
        "seat": seat,
        "market_relevant_steps": sorted(relevant),
        "applied_count": len(runtime.applied),
    }


def run_exact(task):
    game, kind, step, choice, parent_path, opponent_path = task
    records = mutate_records(game, kind, step, choice)
    runtime = ScriptedTrajectoryRuntime(parent_path, records)
    learner = V4HybridRolloutAgent(
        option_policy=runtime,
        min_option_confidence=0.0,
        enable_market_race_ordering=True,
        enable_structural_route_gate=False,
    )
    seat = int(game["seat"])
    seed = int(game["seed"])
    opponent = str(opponent_path)
    agents = [learner, opponent] if seat == 0 else [opponent, learner]
    env = make(
        "kaggriculture",
        configuration={"seed": seed, "episodeSteps": 720},
        debug=False,
    )
    env.run(agents)
    if len(env.steps) != 720:
        raise RuntimeError(f"seed={seed} seat={seat}: {len(env.steps)} frames")
    final = env.steps[-1]
    statuses = [str(x.status) for x in final]
    if statuses != ["DONE", "DONE"]:
        raise RuntimeError(f"seed={seed} seat={seat}: statuses={statuses}")
    farms = final[0].observation.farms
    own = int(farms[seat].money)
    rival = int(farms[1 - seat].money)
    capture = next(
        (
            row for row in runtime.applied
            if int(row["step"]) == int(step)
        ),
        None,
    )
    if capture is None:
        raise RuntimeError(
            f"seed={seed} seat={seat} step={step}: capture missing"
        )
    return {
        "seed": seed,
        "seat": seat,
        "kind": kind,
        "step": int(step),
        "choice": choice,
        "own_money": own,
        "v51_money": rival,
        "margin": own - rival,
        "stored_margin": int(game["margin"]),
        "hidden_f16_hex": capture["hidden_f16"].hex(),
        "clock_f16_hex": capture["clock_f16"].hex(),
        "base_route_class": int(capture["base_route_class"]),
        "route_action": int(capture["route_action"]),
        "market_action": int(capture["market_action"]),
        "horizon_action": int(capture["horizon_action"]),
        "market_mask_bits": int(capture["market_mask_bits"]),
    }
def task_list(
    winners,
    parent_path,
    opponent_path,
    do_market,
    do_horizon,
    market_relevant_by_game=None,
):
    tasks = []
    market_relevant_by_game = market_relevant_by_game or {}
    for game in winners:
        game_key = (int(game["seed"]), int(game["seat"]))
        relevant_steps = market_relevant_by_game.get(game_key)
        for record in game["records"]:
            step = int(record["step"])
            if do_market and (
                relevant_steps is None or step in relevant_steps
            ):
                for mode in MARKETS:
                    tasks.append(
                        (game, "market", step, mode, parent_path, opponent_path)
                    )
            if do_horizon:
                route_changed = (
                    int(record["route_action"]) != int(record["base_route_class"])
                )
                if route_changed:
                    for horizon in HORIZONS:
                        tasks.append(
                            (
                                game,
                                "horizon",
                                step,
                                int(horizon),
                                parent_path,
                                opponent_path,
                            )
                        )
    return tasks


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["seed"], row["seat"], row["kind"], row["step"])
        groups.setdefault(key, []).append(row)
    summary = []
    for key, items in groups.items():
        stored = int(items[0]["stored_margin"])
        ranked = sorted(items, key=lambda x: int(x["margin"]), reverse=True)
        best = ranked[0]
        original = None
        for item in items:
            if item["kind"] == "market":
                # Original choice is filled by caller after lookup.
                if item.get("is_original"):
                    original = item
            elif item["kind"] == "horizon" and item.get("is_original"):
                original = item
        original_margin = (
            int(original["margin"]) if original is not None else stored
        )
        summary.append(
            {
                "seed": key[0],
                "seat": key[1],
                "kind": key[2],
                "step": key[3],
                "stored_margin": stored,
                "best_choice": best["choice"],
                "best_margin": int(best["margin"]),
                "best_gain_vs_stored": int(best["margin"]) - stored,
                "original_choice": original["choice"] if original else "",
                "original_margin": original_margin,
                "best_gain_vs_original": (
                    int(best["margin"]) - original_margin
                ),
                "choice_count": len(items),
            }
        )
    return sorted(
        summary,
        key=lambda x: (
            -int(x["stored_margin"]),
            str(x["kind"]),
            int(x["step"]),
        ),
    )
def mark_original(rows, winners):
    lookup = {}
    for game in winners:
        for record in game["records"]:
            key_m = (
                int(game["seed"]),
                int(game["seat"]),
                "market",
                int(record["step"]),
            )
            lookup[key_m] = str(record["market_mode"])
            key_h = (
                int(game["seed"]),
                int(game["seat"]),
                "horizon",
                int(record["step"]),
            )
            lookup[key_h] = int(record["horizon"])
    for row in rows:
        key = (
            int(row["seed"]),
            int(row["seat"]),
            str(row["kind"]),
            int(row["step"]),
        )
        row["is_original"] = row["choice"] == lookup.get(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--memory",
        default=str(ROOT / "output" / "winner_memory_v4.pt"),
    )
    ap.add_argument(
        "--parent",
        default=str(ROOT / "assets" / "parent_promoted_v2.pt"),
    )
    ap.add_argument(
        "--opponent",
        default=str(ROOT / "assets" / "v51_main.py"),
    )
    ap.add_argument("--top-winners", type=int, default=3)
    ap.add_argument(
        "--winner-offset",
        type=int,
        default=0,
        help="Skip this many strongest winners before selecting top-winners.",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=1,
        help="Recycle exact-game workers to prevent simulator/PyTorch RAM retention.",
    )
    ap.add_argument(
        "--kind",
        choices=("market", "horizon", "both"),
        default="both",
    )
    ap.add_argument(
        "--output",
        default=str(ROOT / "output" / "causal_refine_v4_1.json"),
    )
    args = ap.parse_args()

    winners = load_winners(
        Path(args.memory),
        args.top_winners,
        args.winner_offset,
    )
    if not winners:
        raise RuntimeError("no winning trajectories in memory")
    do_market = args.kind in ("market", "both")
    do_horizon = args.kind in ("horizon", "both")

    market_probes = []
    market_relevant_by_game = {}
    if do_market:
        ctx = mp.get_context("spawn")
        probe_args = [
            (
                game,
                Path(args.parent),
                Path(args.opponent),
            )
            for game in winners
        ]
        with ctx.Pool(processes=1, maxtasksperchild=1) as probe_pool:
            market_probes = probe_pool.starmap(
                probe_market_relevant,
                probe_args,
                chunksize=1,
            )
        for probe in market_probes:
            key = (int(probe["seed"]), int(probe["seat"]))
            market_relevant_by_game[key] = set(
                int(x) for x in probe["market_relevant_steps"]
            )
            print(
                f"[MARKET-PROBE] seed={probe['seed']} "
                f"seat={probe['seat']} relevant="
                f"{len(probe['market_relevant_steps'])}/"
                f"{probe['applied_count']} "
                f"steps={probe['market_relevant_steps']}",
                flush=True,
            )

    tasks = task_list(
        winners,
        Path(args.parent),
        Path(args.opponent),
        do_market,
        do_horizon,
        market_relevant_by_game=market_relevant_by_game,
    )
    print(
        f"CAUSAL tasks={len(tasks)} winners={len(winners)} "
        f"workers={args.workers} kind={args.kind}",
        flush=True,
    )
    ctx = mp.get_context("spawn")
    worker_count = max(1, min(int(args.workers), len(tasks)))
    recycle_after = max(1, int(args.max_tasks_per_child))
    with ctx.Pool(
        processes=worker_count,
        maxtasksperchild=recycle_after,
    ) as pool:
        rows = []
        for i, row in enumerate(
            pool.imap_unordered(run_exact, tasks, chunksize=1),
            1,
        ):
            rows.append(row)
            if i % worker_count == 0 or i == len(tasks):
                print(f"[CAUSAL] {i}/{len(tasks)}", flush=True)

    mark_original(rows, winners)
    summary = summarize(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "farmos_v51_causal_refine_v4_1",
        "winner_count": len(winners),
        "task_count": len(tasks),
        "market_probes": market_probes,
        "winners": [
            {
                "seed": int(g["seed"]),
                "seat": int(g["seat"]),
                "margin": int(g["margin"]),
                "own_money": int(g["own_money"]),
                "v51_money": int(g["v51_money"]),
            }
            for g in winners
        ],
        "summary": summary,
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    if summary:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    improvements = [
        x for x in summary if int(x["best_gain_vs_original"]) > 0
    ]
    gains = [int(x["best_gain_vs_original"]) for x in improvements]
    print(
        f"CAUSAL_DONE groups={len(summary)} improved={len(improvements)} "
        f"median_positive_gain="
        f"{statistics.median(gains) if gains else 0:+.1f} "
        f"max_gain={max(gains) if gains else 0:+d}",
        flush=True,
    )
    for item in sorted(
        improvements,
        key=lambda x: int(x["best_gain_vs_original"]),
        reverse=True,
    )[:20]:
        print(
            f"[GAIN] seed={item['seed']} seat={item['seat']} "
            f"{item['kind']} step={item['step']} "
            f"{item['original_choice']}->{item['best_choice']} "
            f"gain={int(item['best_gain_vs_original']):+d} "
            f"margin={int(item['best_margin']):+d}",
            flush=True,
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()
