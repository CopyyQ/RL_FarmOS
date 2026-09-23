import argparse
import copy
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT), str(ROOT / "src")]

from kaggle_environments import make
from continuous_runtime import ScriptedTrajectoryRuntime, HORIZONS
from rollout.v4_hybrid_agent import V4HybridRolloutAgent
from structural_refine import choose_tasks
from kaggrl.v45_macro_data import load_v45_macro_data


DEFAULT_STEPS = (144, 168, 192, 216, 240, 264, 288)


def _run_exact(task):
    (
        game,
        target_step,
        route_id,
        horizon,
        parent_path,
        opponent_path,
        route_to_class,
    ) = task
    records = copy.deepcopy(game["records"])
    is_baseline = int(target_step) < 0
    if not is_baseline:
        found = False
        for record in records:
            if int(record["step"]) != int(target_step):
                continue
            found = True
            record["chosen_route_id"] = int(route_id)
            record["route_action"] = int(route_to_class[int(route_id)])
            record["horizon"] = int(horizon)
            record["horizon_action"] = int(HORIZONS.index(int(horizon)))
            break
        if not found:
            raise KeyError(f"missing target step {target_step}")

    runtime = ScriptedTrajectoryRuntime(parent_path, records)
    learner = V4HybridRolloutAgent(
        option_policy=runtime,
        min_option_confidence=0.0,
        enable_market_race_ordering=True,
        enable_idle_weed_labor=True,
        allow_all_routes=True,
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
    farms = env.steps[-1][0].observation.farms
    own = int(farms[seat].money)
    rival = int(farms[1 - seat].money)

    state_row = None
    if not is_baseline:
        for applied in runtime.applied:
            if int(applied["step"]) == int(target_step):
                state_row = dict(applied)
                break

    return {
        "seed": seed,
        "seat": seat,
        "target_step": int(target_step),
        "route_id": int(route_id),
        "horizon": int(horizon),
        "margin": own - rival,
        "own_money": own,
        "v51_money": rival,
        "state": state_row,
        "mismatches": list(runtime.mismatches),
    }


def _candidate_plan(winners, route_ids, route_to_class, steps, per_step):
    routes, _, _ = load_v45_macro_data()
    raw, meta = choose_tasks(
        winners,
        routes,
        route_ids,
        route_to_class,
        top_per_direction=2,
    )
    grouped = defaultdict(list)
    for task, info in zip(raw, meta):
        if int(info["step"]) not in steps:
            continue
        key = (int(info["seed"]), int(info["seat"]), int(info["step"]))
        grouped[key].append((task, info))

    selected = []
    for key, items in grouped.items():
        chosen = []
        seen = set()

        def add(item):
            task, info = item
            sig = (int(info["route"]), int(info["horizon"]))
            if sig in seen:
                return
            seen.add(sig)
            chosen.append((task, info))

        positives = sorted(
            [x for x in items if float(x[1]["structural_delta"]) > 0],
            key=lambda x: (
                float(x[1]["structural_delta"]),
                float(x[1]["distance"]),
            ),
            reverse=True,
        )
        negatives = sorted(
            [x for x in items if float(x[1]["structural_delta"]) < 0],
            key=lambda x: (
                float(x[1]["structural_delta"]),
                -float(x[1]["distance"]),
            ),
        )
        diverse = sorted(
            items,
            key=lambda x: float(x[1]["distance"]),
            reverse=True,
        )
        for pool in (positives, negatives, diverse):
            if pool:
                add(pool[0])

        # Preserve the discovered high-value early cow branch explicitly.
        if key[2] == 144:
            for item in items:
                info = item[1]
                if int(info["route"]) == 105 and int(info["horizon"]) in (48, 72):
                    add(item)

        for item in chosen[: max(1, int(per_step))]:
            selected.append(item)
    return selected
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--memory",
        default=str(ROOT / "output" / "winner_memory_v4.pt"),
    )
    ap.add_argument("--top-winners", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-step", type=int, default=4)
    ap.add_argument(
        "--steps",
        default=",".join(str(x) for x in DEFAULT_STEPS),
    )
    ap.add_argument(
        "--output",
        default=str(ROOT / "output" / "structural_causal_v4_2.pt"),
    )
    args = ap.parse_args()

    steps = {int(x) for x in args.steps.split(",") if x.strip()}
    memory = torch.load(args.memory, map_location="cpu", weights_only=False)
    winners = sorted(
        [g for g in memory["games"] if int(g["margin"]) > 0],
        key=lambda g: int(g["margin"]),
        reverse=True,
    )[: int(args.top_winners)]

    parent_payload = torch.load(
        ROOT / "assets" / "parent_promoted_v2.pt",
        map_location="cpu",
        weights_only=False,
    )
    route_ids = tuple(int(x) for x in parent_payload["route_ids"])
    route_to_class = {route: idx for idx, route in enumerate(route_ids)}
    selected = _candidate_plan(
        winners,
        route_ids,
        route_to_class,
        steps,
        args.per_step,
    )

    parent_path = ROOT / "assets" / "parent_promoted_v2.pt"
    opponent_path = ROOT / "assets" / "v51_main.py"

    baseline_tasks = [
        (
            game,
            -1,
            -1,
            1,
            parent_path,
            opponent_path,
            route_to_class,
        )
        for game in winners
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(args.workers, len(baseline_tasks))) as pool:
        baselines = list(pool.imap_unordered(_run_exact, baseline_tasks))

    baseline_map = {
        (int(row["seed"]), int(row["seat"])): int(row["margin"])
        for row in baselines
    }
    print(f"STRUCT_BASELINES {baseline_map}", flush=True)

    metadata_map = {}
    tasks = []
    for task, info in selected:
        game, step, route, horizon = task
        full = (
            game,
            step,
            route,
            horizon,
            parent_path,
            opponent_path,
            route_to_class,
        )
        tasks.append(full)
        key = (
            int(game["seed"]),
            int(game["seat"]),
            int(step),
            int(route),
            int(horizon),
        )
        metadata_map[key] = info

    print(
        f"STRUCT_COLLECT candidates={len(tasks)} winners={len(winners)} "
        f"steps={sorted(steps)} workers={args.workers}",
        flush=True,
    )

    rows = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ctx.Pool(processes=args.workers) as pool:
        for i, row in enumerate(
            pool.imap_unordered(_run_exact, tasks, chunksize=1),
            1,
        ):
            key = (
                int(row["seed"]),
                int(row["seat"]),
                int(row["target_step"]),
                int(row["route_id"]),
                int(row["horizon"]),
            )
            info = metadata_map[key]
            row["baseline_margin"] = baseline_map[(row["seed"], row["seat"])]
            row["gain"] = int(row["margin"]) - int(row["baseline_margin"])
            row["original_route"] = int(info["original_route"])
            row["structural_delta"] = float(info["structural_delta"])
            row["distance"] = float(info["distance"])
            row["base_profile"] = dict(info["base_profile"])
            row["alt_profile"] = dict(info["alt_profile"])
            rows.append(row)

            if i % max(1, args.workers) == 0 or i == len(tasks):
                torch.save(
                    {
                        "schema": "farmos_structural_causal_v4_2_partial",
                        "baselines": baselines,
                        "rows": rows,
                        "completed": i,
                        "total": len(tasks),
                        "route_ids": route_ids,
                        "steps": sorted(steps),
                    },
                    output,
                )
                print(f"[STRUCT_COLLECT] {i}/{len(tasks)}", flush=True)

    usable = [
        row for row in rows
        if isinstance(row.get("state"), dict)
        and not row.get("mismatches")
    ]
    payload = {
        "schema": "farmos_structural_causal_v4_2",
        "baselines": baselines,
        "rows": rows,
        "usable_rows": usable,
        "route_ids": route_ids,
        "steps": sorted(steps),
        "winner_keys": [
            (int(g["seed"]), int(g["seat"])) for g in winners
        ],
    }
    torch.save(payload, output)

    print(
        f"STRUCT_COLLECT_DONE rows={len(rows)} usable={len(usable)}",
        flush=True,
    )
    for row in sorted(usable, key=lambda x: int(x["gain"]), reverse=True)[:20]:
        print(
            f"[STRUCT_GAIN] seed={row['seed']} seat={row['seat']} "
            f"step={row['target_step']} route={row['original_route']}"
            f"->{row['route_id']} h={row['horizon']} "
            f"gain={int(row['gain']):+d} margin={int(row['margin']):+d}",
            flush=True,
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()
