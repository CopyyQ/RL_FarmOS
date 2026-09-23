from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import random
import signal
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Only the main process handles Ctrl+C. Spawned rollout workers exit when the
# parent terminates the persistent pool, without noisy KeyboardInterrupt traces.
if mp.current_process().name != "MainProcess":
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

from continuous_runtime import (
    ContinuousActor,
    ContinuousRuntime,
    ScriptedTrajectoryRuntime,
    HORIZONS,
    ECONOMIC_DIM,
    OPPONENT_DIM,
    HARVEST_ECON_DIM,
    init_actor,
    load_actor_state_compatible,
    load_parent,
)
from kaggrl.v4_options import MARKET_MODES
from kaggrl.v4_farm_supervisor import (
    farm_transition_reward,
    MICRO_TASKS,
    MICRO_LOCAL_DIM,
    PLANT_CONTEXT_DIM,
)
from kaggrl.v4_structural_gate import (
    COW_BRANCH_SOURCE_ROUTES,
    STRUCTURAL_GAIN_TABLE,
)
from rollout.v4_hybrid_agent import V4HybridRolloutAgent
from v45_skill_runtime_actkeep import V45SkillActKeepRuntime as V45SkillRuntime


ALLOWED_MARKETS = (
    "KEEP_ROUTE",
    "LIQUIDATE_SHED",
    "HOLD_SALES",
    "FRONT_RUN_1",
    "FRONT_RUN_9",
)


def parse_seed_range(text: str):
    if ":" in text:
        start, count = [int(x) for x in text.split(":", 1)]
        return list(range(start, start + count))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def terminal_reward(own_money: float, rival_money: float) -> float:
    margin = float(own_money - rival_money)
    if margin > 0:
        win = 25.0
    elif margin < 0:
        win = -25.0
    else:
        win = 0.0
    margin_term = 10.0 * math.tanh(margin / 9000.0)
    money_term = 2.0 * math.tanh((float(own_money) - 70000.0) / 35000.0)
    return win + margin_term + money_term


def attach_dense_farm_rewards(env_steps, seat, records):
    step_rewards = {}
    totals = {
        "dense_reward": 0.0, "plant_deaths": 0, "animal_escapes": 0,
        "watered": 0, "fed": 0, "cared": 0, "fertilized": 0,
        "fertilizer_collected": 0, "harvested_units": 0,
        "placed_units": 0, "sold_units": 0,
        "plants_created": 0, "structures_built": 0, "animals_placed": 0,
        "yield_gain": 0, "invalid_ops": 0,
        "critical_water_opportunities": 0, "water_attempts": 0, "water_success_actions": 0,
        "fert_collect_opportunities": 0, "fert_collect_attempts": 0, "fert_collect_success_actions": 0,
    }
    for i in range(max(0, len(env_steps) - 1)):
        prev_state = env_steps[i][seat]
        curr_state = env_steps[i + 1][seat]
        # kaggle-environments stores the action that produced state i+1 on
        # env_steps[i+1], not env_steps[i]. Align action-conditioned shaping
        # with the actual transition i -> i+1.
        reward, stats = farm_transition_reward(
            prev_state.observation,
            curr_state.observation,
            getattr(curr_state, "action", None) or {},
            seat,
        )
        step_rewards[i] = float(reward)
        totals["dense_reward"] += float(reward)
        for key in totals:
            if key != "dense_reward" and key in stats:
                totals[key] += int(stats[key])

    ordered = sorted(records, key=lambda r: int(r.get("step", 0)))
    for index, record in enumerate(ordered):
        start = int(record.get("step", 0))
        end = (
            int(ordered[index + 1].get("step", start + 1))
            if index + 1 < len(ordered)
            else len(env_steps) - 1
        )
        record["dense_reward"] = float(
            sum(step_rewards.get(step, 0.0) for step in range(start, end))
        )
    return totals


def attach_micro_rewards(env_steps, seat, micro_records, terminal):
    if not micro_records:
        return []
    step_reward = {}
    for i in range(max(0, len(env_steps) - 1)):
        reward, _ = farm_transition_reward(
            env_steps[i][seat].observation,
            env_steps[i + 1][seat].observation,
            getattr(env_steps[i + 1][seat], "action", None) or {},
            seat,
        )
        step_reward[i] = float(reward)
    groups = {}
    for record in micro_records:
        groups.setdefault(int(record.get("hand_idx", 0)), []).append(record)
    active_hands = max(1, len(groups))
    for hand_rows in groups.values():
        hand_rows.sort(key=lambda r: int(r.get("step", 0)))
        for index, record in enumerate(hand_rows):
            start = int(record.get("step", 0))
            day_end = ((start // 24) + 1) * 24
            end = (
                int(hand_rows[index + 1].get("step", start + 1))
                if index + 1 < len(hand_rows)
                else min(day_end, len(env_steps) - 1)
            )
            dense = sum(step_reward.get(s, 0.0) for s in range(start, end))
            record["micro_reward"] = float(max(-1.0, min(1.0, 0.25 * dense)))
        # Micro credit is primarily operational. Keep terminal money/win as a
        # weak tie-breaker so idle-labor learning cannot simply piggyback on a
        # strong macro trajectory.
        hand_rows[-1]["micro_reward"] += float(0.02 * terminal / active_hands)
    return micro_records


def _run_chunk(args):
    # Backward compatibility for winner reproduction helpers that still pass
    # the pre-V4.5 10-field rollout tuple.
    if len(args) == 10:
        args = tuple(args) + (720, 4.50, 432, 0.70)
    (
        parent_path,
        snapshot_path,
        opponent_path,
        games,
        stochastic,
        temperature,
        residual_scale,
        base_keep_bias,
        decision_every,
        iteration,
        skill_cutover_step,
        skill_keep_penalty,
        skill_shadow_start_step,
        skill_confidence_threshold,
    ) = args

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    # kaggle-environments emits noisy optional OpenSpiel registration messages
    # on some installs. Silence only the import; game errors remain visible.
    saved_out, saved_err = os.dup(1), os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        from kaggle_environments import make
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(null_fd)

    runtime = V45SkillRuntime(
        parent_path,
        snapshot_path,
        stochastic=stochastic,
        temperature=temperature,
        residual_scale=residual_scale,
        base_keep_bias=base_keep_bias,
        decision_every=decision_every,
        allowed_market_modes=ALLOWED_MARKETS,
        skill_cutover_step=skill_cutover_step,
        skill_keep_penalty=skill_keep_penalty,
        skill_shadow_start_step=skill_shadow_start_step,
        skill_confidence_threshold=skill_confidence_threshold,
        seed=iteration * 1_000_003 + 17,
    )
    learner = V4HybridRolloutAgent(
        option_policy=runtime,
        min_option_confidence=0.0,
        enable_market_race_ordering=True,
    )
    out = []
    for seed, seat in games:
        runtime.reset()
        runtime.rng.manual_seed(
            int(seed) * 1009 + int(seat) * 97 + int(iteration) * 1_000_003
        )
        learner.reset()
        agents = [learner, opponent_path] if seat == 0 else [opponent_path, learner]
        env = make(
            "kaggriculture",
            configuration={"seed": int(seed), "episodeSteps": 720},
            debug=False,
        )
        env.run(agents)
        if len(env.steps) != 720:
            raise RuntimeError(
                f"seed={seed} seat={seat}: expected 720 frames, got {len(env.steps)}"
            )
        final = env.steps[-1]
        statuses = [str(x.status) for x in final]
        if statuses != ["DONE", "DONE"]:
            raise RuntimeError(
                f"seed={seed} seat={seat}: invalid final status {statuses}"
            )
        farms = final[0].observation.farms
        own = int(farms[seat].money)
        rival = int(farms[1 - seat].money)
        margin = own - rival
        dense_stats = attach_dense_farm_rewards(
            env.steps, seat, runtime.records
        )
        terminal = terminal_reward(own, rival)
        micro_records = attach_micro_rewards(
            env.steps, seat, runtime.micro_records, terminal
        )
        out.append(
            {
                "seed": int(seed),
                "seat": int(seat),
                "own_money": own,
                "v51_money": rival,
                "margin": margin,
                "win": int(margin > 0),
                "reward": terminal,
                "dense_stats": dense_stats,
                "records": list(runtime.records),
                "micro_records": list(micro_records),
                "skill_teacher_records": list(
                    getattr(runtime, "teacher_records", [])
                ),
                "skill_exec_trace": list(
                    getattr(runtime, "skill_exec_trace", [])
                ),
                "skill_stats": dict(
                    getattr(runtime, "skill_stats", {}) or {}
                ),
            }
        )
    return out


def chunks(items, n):
    n = max(1, min(int(n), len(items)))
    result = [[] for _ in range(n)]
    for i, item in enumerate(items):
        result[i % n].append(item)
    return [x for x in result if x]


def exact_games(
    parent_path,
    snapshot_path,
    opponent_path,
    seeds,
    workers,
    stochastic,
    temperature,
    residual_scale,
    base_keep_bias,
    decision_every,
    iteration,
    *,
    pool=None,
    both_seats=True,
    skill_cutover_step=720,
    skill_keep_penalty=4.50,
    skill_shadow_start_step=0,
    skill_confidence_threshold=0.70,
):
    if both_seats:
        games = [(int(seed), seat) for seed in seeds for seat in (0, 1)]
    else:
        # One training seat per fresh seed; parity flips each iteration so both
        # seats are still explored over time while halving rollout cost.
        games = [
            (int(seed), (int(seed) + int(iteration)) & 1)
            for seed in seeds
        ]
    parts = chunks(games, workers)
    tasks = [
        (
            str(parent_path),
            str(snapshot_path),
            str(opponent_path),
            part,
            bool(stochastic),
            float(temperature),
            float(residual_scale),
            float(base_keep_bias),
            int(decision_every),
            int(iteration),
            int(skill_cutover_step),
            float(skill_keep_penalty),
            int(skill_shadow_start_step),
            float(skill_confidence_threshold),
        )
        for part in parts
    ]
    rows = []
    if len(tasks) == 1:
        return _run_chunk(tasks[0])
    if pool is not None:
        results = pool.map(_run_chunk, tasks)
    else:
        ctx = mp.get_context("spawn")
        local_pool = ctx.Pool(processes=len(tasks))
        try:
            results = local_pool.map(_run_chunk, tasks)
            local_pool.close()
            local_pool.join()
        except BaseException:
            local_pool.terminate()
            local_pool.join()
            raise
    for part in results:
        rows.extend(part)
    rows.sort(key=lambda x: (x["seed"], x["seat"]))
    return rows


def metrics(rows):
    margins = [x["margin"] for x in rows]
    own = [x["own_money"] for x in rows]
    rival = [x["v51_money"] for x in rows]
    wins = sum(x["win"] for x in rows)
    return {
        "games": len(rows),
        "wins": wins,
        "win_rate": wins / max(1, len(rows)),
        "own_money": statistics.fmean(own),
        "v51_money": statistics.fmean(rival),
        "mean_margin": statistics.fmean(margins),
        "median_margin": statistics.median(margins),
        "min_margin": min(margins),
        "max_margin": max(margins),
        "mean_reward": statistics.fmean(x["reward"] for x in rows),
        "decisions": sum(len(x["records"]) for x in rows),
        "shop_replans": sum(
            int(bool(r.get("shop_replan", False)))
            for x in rows for r in x["records"]
        ),
        "handoff_candidates_mean": statistics.fmean(
            int(r.get("handoff_candidates", 1))
            for x in rows for r in x["records"]
        ) if any(x.get("records") for x in rows) else 1.0,
        "handoff_locked": sum(
            int(r.get("handoff_candidates", 1)) <= 1
            for x in rows for r in x["records"]
        ),
        "dense_reward": statistics.fmean(
            float((x.get("dense_stats") or {}).get("dense_reward", 0.0))
            for x in rows
        ),
        "plant_deaths": sum(
            int((x.get("dense_stats") or {}).get("plant_deaths", 0)) for x in rows
        ),
        "animal_escapes": sum(
            int((x.get("dense_stats") or {}).get("animal_escapes", 0)) for x in rows
        ),
        "watered": sum(int((x.get("dense_stats") or {}).get("watered", 0)) for x in rows),
        "fed": sum(int((x.get("dense_stats") or {}).get("fed", 0)) for x in rows),
        "cared": sum(int((x.get("dense_stats") or {}).get("cared", 0)) for x in rows),
        "fertilized": sum(int((x.get("dense_stats") or {}).get("fertilized", 0)) for x in rows),
        "fertilizer_collected": sum(int((x.get("dense_stats") or {}).get("fertilizer_collected", 0)) for x in rows),
        "harvested_units": sum(int((x.get("dense_stats") or {}).get("harvested_units", 0)) for x in rows),
        "placed_units": sum(int((x.get("dense_stats") or {}).get("placed_units", 0)) for x in rows),
        "sold_units": sum(int((x.get("dense_stats") or {}).get("sold_units", 0)) for x in rows),
        "plants_created": sum(int((x.get("dense_stats") or {}).get("plants_created", 0)) for x in rows),
        "structures_built": sum(int((x.get("dense_stats") or {}).get("structures_built", 0)) for x in rows),
        "animals_placed": sum(int((x.get("dense_stats") or {}).get("animals_placed", 0)) for x in rows),
        "micro_records": sum(len(x.get("micro_records", [])) for x in rows),
        "skill_teacher_records": sum(
            len(x.get("skill_teacher_records", [])) for x in rows
        ),
        "invalid_ops": sum(int((x.get("dense_stats") or {}).get("invalid_ops", 0)) for x in rows),
        "critical_water_opportunities": sum(int((x.get("dense_stats") or {}).get("critical_water_opportunities", 0)) for x in rows),
        "water_attempts": sum(int((x.get("dense_stats") or {}).get("water_attempts", 0)) for x in rows),
        "water_success_actions": sum(int((x.get("dense_stats") or {}).get("water_success_actions", 0)) for x in rows),
        "fert_collect_opportunities": sum(int((x.get("dense_stats") or {}).get("fert_collect_opportunities", 0)) for x in rows),
        "fert_collect_attempts": sum(int((x.get("dense_stats") or {}).get("fert_collect_attempts", 0)) for x in rows),
        "fert_collect_success_actions": sum(int((x.get("dense_stats") or {}).get("fert_collect_success_actions", 0)) for x in rows),
        "micro_decisions": sum(len(x.get("micro_records") or []) for x in rows),
        "micro_nonkeep": sum(
            int(r.get("task_action", 0)) != 0
            for x in rows for r in (x.get("micro_records") or [])
        ),
        "skill_records": sum(
            int(bool(r.get("skill_mode", False)))
            for x in rows for r in (x.get("micro_records") or [])
        ),
        "skill_nonkeep": sum(
            int(bool(r.get("skill_mode", False)))
            and int(r.get("task_action", 0)) != 0
            for x in rows for r in (x.get("micro_records") or [])
        ),
        "skill_samples": sum(
            int((x.get("skill_stats") or {}).get("samples", 0)) for x in rows
        ),
        "skill_keep_samples": sum(
            int((x.get("skill_stats") or {}).get("keep_samples", 0)) for x in rows
        ),
        "skill_nonkeep_proposals": sum(
            int((x.get("skill_stats") or {}).get("nonkeep_proposals", 0)) for x in rows
        ),
        "skill_confident_nonkeep": sum(
            int((x.get("skill_stats") or {}).get("confident_nonkeep", 0)) for x in rows
        ),
        "skill_semantic_safe_nonkeep": sum(
            int((x.get("skill_stats") or {}).get("semantic_safe_nonkeep", 0)) for x in rows
        ),
        "skill_executed_sampled": sum(
            int((x.get("skill_stats") or {}).get("executed_sampled", 0)) for x in rows
        ),
        "skill_executed_actions": sum(
            int((x.get("skill_stats") or {}).get("executed_actions", 0)) for x in rows
        ),
    }


def print_metrics(tag, iteration, m, best=None, elapsed=None):
    extra = ""
    if best is not None:
        extra += f" best_margin={best:+.1f}"
    if elapsed is not None:
        games_per_sec = m["games"] / max(elapsed, 1e-9)
        frames_per_sec = m["games"] * 720.0 / max(elapsed, 1e-9)
        extra += (
            f" elapsed={elapsed:.1f}s"
            f" throughput={games_per_sec:.3f}g/s"
            f" frames={frames_per_sec:.0f}/s"
        )
    print(
        f"[{tag}] iter={iteration:05d} games={m['games']:3d} "
        f"WIN={m['wins']:3d}/{m['games']:3d} ({100*m['win_rate']:6.2f}%) "
        f"money={m['own_money']:10.1f} v51={m['v51_money']:10.1f} "
        f"margin={m['mean_margin']:+10.1f} median={m['median_margin']:+9.1f} "
        f"range=[{m['min_margin']:+d},{m['max_margin']:+d}] "
        f"decisions={m['decisions']:4d} replans={m.get('shop_replans', 0):3d}{extra}",
        flush=True,
    )


def economic_rollout_summary(rows):
    contexts = []
    for row in rows:
        context = _game_economic_context(row)
        if context is not None:
            contexts.append(context)
    if not contexts:
        return None
    mean = np.mean(np.stack(contexts, axis=0), axis=0)
    crop_names = ("wheat", "carrot", "tomato", "strawberry", "melon")
    crop_scores = mean[38:43]
    crop_best = crop_names[int(np.argmax(crop_scores))]
    return {
        "yarn_shops": float(mean[3] * 8.0),
        "pizza_shops": float(mean[1] * 8.0),
        "goose_roi": float(mean[35]),
        "cow_roi": float(mean[36]),
        "sheep_roi": float(mean[37]),
        "crop_best": crop_best,
        "crop_best_score": float(np.max(crop_scores)),
        "context_games": len(contexts),
    }


def run_scripted_reproduction(game, parent_path, opponent_path):
    from kaggle_environments import make

    runtime = ScriptedTrajectoryRuntime(parent_path, game["records"])
    learner = V4HybridRolloutAgent(
        option_policy=runtime,
        min_option_confidence=0.0,
        enable_market_race_ordering=True,
        enable_structural_route_gate=False,
    )
    seat = int(game["seat"])
    seed = int(game["seed"])
    opponent_agent = str(opponent_path)
    agents = (
        [learner, opponent_agent]
        if seat == 0
        else [opponent_agent, learner]
    )
    env = make(
        "kaggriculture",
        configuration={"seed": seed, "episodeSteps": 720},
        debug=False,
    )
    env.run(agents)
    if len(env.steps) != 720:
        raise RuntimeError(
            f"scripted reproduction seed={seed} seat={seat}: "
            f"expected 720 frames, got {len(env.steps)}"
        )
    final = env.steps[-1]
    statuses = [str(x.status) for x in final]
    if statuses != ["DONE", "DONE"]:
        raise RuntimeError(
            f"scripted reproduction seed={seed} seat={seat}: "
            f"invalid final status {statuses}"
        )
    farms = final[0].observation.farms
    own = int(farms[seat].money)
    rival = int(farms[1 - seat].money)
    margin = own - rival
    return {
        "seed": seed,
        "seat": seat,
        "own_money": own,
        "v51_money": rival,
        "margin": margin,
        "win": int(margin > 0),
        "mismatches": list(runtime.mismatches),
        "applied_decisions": len(runtime.applied),
        "expected_margin": int(game["margin"]),
        "exact_margin_match": margin == int(game["margin"]),
    }


def run_actor_reproduction(
    game,
    parent_path,
    snapshot_path,
    opponent_path,
    *,
    temperature,
    residual_scale,
    base_keep_bias,
    decision_every,
    iteration,
):
    task = (
        str(parent_path),
        str(snapshot_path),
        str(opponent_path),
        [(int(game["seed"]), int(game["seat"]))],
        False,
        float(temperature),
        float(residual_scale),
        float(base_keep_bias),
        int(decision_every),
        int(iteration),
    )
    return _run_chunk(task)[0]


def pick_verified_winner(games, parent_path, opponent_path, max_checks=3):
    winners = partition_elite_games(games)["winner"]
    failures = []
    for game in winners[: max(1, int(max_checks))]:
        replay = run_scripted_reproduction(game, parent_path, opponent_path)
        # V4.2 intentionally changes deterministic low-level execution
        # (market-race ordering, idle weed labor). Historical elite margins
        # therefore need not byte-match the margin produced by the current
        # runtime. Verification is exact with respect to the CURRENT runtime:
        # the full 720-step replay must still win and the strategic trace must
        # apply without route/base mismatches.
        valid = (
            replay["win"] == 1
            and not replay["mismatches"]
        )
        if valid:
            return game, replay, failures
        failures.append(replay)
    return None, None, failures


def amplify_single_winner(
    actor,
    parent,
    winner_optimizer,
    harvest_optimizer,
    elite_games,
    winner_game,
    snapshot_path,
    parent_path,
    opponent_path,
    *,
    hidden_dim,
    clock_dim,
    route_count,
    market_count,
    device,
    residual_scale,
    base_keep_bias,
    decision_every,
    temperature,
    epochs,
    rounds,
    minibatch,
    iteration,
):
    batch = winner_records_to_batch(
        elite_games,
        hidden_dim,
        clock_dim,
        device,
        max_winners=1,
        max_near=0,
        max_good=0,
        winner_only=True,
    )
    if batch is None or batch["winner_rows"] <= 0:
        return {
            "reproduced": False,
            "reason": "no_winner_rows",
            "rounds": 0,
            "actor_margin": None,
            "fit": None,
        }

    save_snapshot(
        actor,
        snapshot_path,
        {"iteration": iteration, "purpose": "winner_reproduction_precheck"},
    )
    actor_repro = run_actor_reproduction(
        winner_game,
        parent_path,
        snapshot_path,
        opponent_path,
        temperature=temperature,
        residual_scale=residual_scale,
        base_keep_bias=base_keep_bias,
        decision_every=decision_every,
        iteration=iteration,
    )
    if actor_repro["margin"] > 0:
        return {
            "reproduced": True,
            "reason": "already_reproduced",
            "rounds": 0,
            "actor_margin": int(actor_repro["margin"]),
            "actor_game": actor_repro,
            "fit": None,
        }

    last_fit = None
    for round_idx in range(1, max(1, int(rounds)) + 1):
        last_fit = winner_bc_update(
            actor,
            parent,
            batch,
            winner_optimizer,
            harvest_optimizer,
            route_count,
            market_count,
            residual_scale=residual_scale,
            base_keep_bias=base_keep_bias,
            epochs=epochs,
            minibatch=minibatch,
        )
        save_snapshot(
            actor,
            snapshot_path,
            {
                "iteration": iteration,
                "purpose": "winner_reproduction",
                "round": round_idx,
            },
        )
        actor_repro = run_actor_reproduction(
            winner_game,
            parent_path,
            snapshot_path,
            opponent_path,
            temperature=temperature,
            residual_scale=residual_scale,
            base_keep_bias=base_keep_bias,
            decision_every=decision_every,
            iteration=iteration + round_idx,
        )
        print(
            f"[WINNER-AMP] iter={iteration:05d} round={round_idx} "
            f"target_seed={winner_game['seed']} seat={winner_game['seat']} "
            f"teacher_margin={int(winner_game['margin']):+d} "
            f"actor_margin={int(actor_repro['margin']):+d} "
            f"route={100*last_fit['route_acc']:.1f}% "
            f"market={100*last_fit['market_acc']:.1f}% "
            f"horizon={100*last_fit['horizon_acc']:.1f}% "
            f"p=({last_fit['winner_route_prob']:.3f},"
            f"{last_fit['winner_market_prob']:.3f},"
            f"{last_fit['winner_horizon_prob']:.3f})",
            flush=True,
        )
        if actor_repro["margin"] > 0:
            return {
                "reproduced": True,
                "reason": "amplified",
                "rounds": round_idx,
                "actor_margin": int(actor_repro["margin"]),
                "actor_game": actor_repro,
                "fit": last_fit,
            }

    return {
        "reproduced": False,
        "reason": "amplification_exhausted",
        "rounds": max(1, int(rounds)),
        "actor_margin": int(actor_repro["margin"]),
        "actor_game": actor_repro,
        "fit": last_fit,
    }


def save_snapshot(actor, path: Path, meta=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "farmos_continuous_v51_actor_v2",
            "actor_state": {
                k: v.detach().cpu() for k, v in actor.state_dict().items()
            },
            "meta": dict(meta or {}),
        },
        path,
    )


def load_v1_warmstart(actor, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # V2/V3/V4 checkpoints can seed V4.1. New conditional heads are
    # left at their safe initialization when absent from an older checkpoint.
    src_actor = payload.get("actor_state")
    if isinstance(src_actor, dict):
        info = load_actor_state_compatible(actor, src_actor)
        return info["matched"] > 0

    # V1 best-response head can still seed compatible layers.
    src = payload.get("gpu_best_response_head")
    if not isinstance(src, dict):
        return False
    current = actor.state_dict()
    mapped = {}
    for key, value in src.items():
        target = key
        if key.startswith("route."):
            target = "route_residual." + key.split(".", 1)[1]
        if target in current and current[target].shape == value.shape:
            mapped[target] = value
    actor.load_state_dict(mapped, strict=False)
    return bool(mapped)


def bitmask_to_mask(bits: torch.Tensor, width: int):
    shifts = torch.arange(width, device=bits.device, dtype=torch.int64)
    return ((bits[:, None] >> shifts[None, :]) & 1).bool()


def _record_economic(r):
    raw = r.get("economic_f16")
    if raw is None:
        return np.zeros(ECONOMIC_DIM, dtype=np.float32)
    values = np.frombuffer(raw, np.float16)
    if values.size != ECONOMIC_DIM:
        raise ValueError(
            f"economic trajectory width mismatch: {values.size} != {ECONOMIC_DIM}"
        )
    return values.astype(np.float32, copy=False)


def _record_harvest(r):
    raw = r.get("harvest_f16")
    if raw is None:
        return np.zeros(HARVEST_ECON_DIM, dtype=np.float32)
    values = np.frombuffer(raw, np.float16).astype(np.float32, copy=False)
    if values.size == HARVEST_ECON_DIM:
        return values
    # V4.6 widened the explicit economics vector from 16D to 34D. The first
    # 16 slots preserve ROI/viability semantics, so historical trajectories can
    # be replayed safely by zero-padding the new time/competition/capacity tail.
    if 0 < values.size < HARVEST_ECON_DIM:
        out = np.zeros(HARVEST_ECON_DIM, dtype=np.float32)
        out[: values.size] = values
        return out
    raise ValueError(
        f"harvest trajectory width mismatch: {values.size} > {HARVEST_ECON_DIM}"
    )


def _record_opponent(r):
    raw = r.get("opponent_f16")
    if raw is None:
        return np.zeros(OPPONENT_DIM, dtype=np.float32)
    values = np.frombuffer(raw, np.float16)
    if values.size != OPPONENT_DIM:
        raise ValueError(
            f"opponent trajectory width mismatch: {values.size} != {OPPONENT_DIM}"
        )
    return values.astype(np.float32, copy=False)


def _record_plant_context(r):
    raw = r.get("plant_ctx_f16")
    if raw is None:
        return np.zeros(PLANT_CONTEXT_DIM, dtype=np.float32)
    values = np.frombuffer(raw, np.float16)
    if values.size != PLANT_CONTEXT_DIM:
        raise ValueError(
            f"plant context width mismatch: {values.size} != {PLANT_CONTEXT_DIM}"
        )
    return values.astype(np.float32, copy=False)


def _game_economic_context(game):
    contexts = [
        _record_economic(record)
        for record in game.get("records", [])
        if record.get("economic_f16") is not None
    ]
    if not contexts:
        return None
    # Shops unlock progressively; averaging the strategic decision contexts
    # captures the actual market regime without leaking it into any individual
    # policy observation.
    return np.mean(np.stack(contexts, axis=0), axis=0).astype(np.float32)


def _context_similarity(game, reference_contexts):
    context = _game_economic_context(game)
    if context is None or not reference_contexts:
        return 0.25
    distance = min(
        float(np.mean(np.abs(context - ref)))
        for ref in reference_contexts
    )
    return float(math.exp(-distance / 0.20))


def records_to_training(rows, hidden_dim, clock_dim, device, gamma, lam):
    packed = []
    all_adv = []
    for game in rows:
        recs = game["records"]
        if not recs:
            continue
        n = len(recs)
        values = np.asarray([r["old_value"] for r in recs], dtype=np.float32)
        rewards = np.asarray(
            [float(r.get("dense_reward", 0.0)) for r in recs],
            dtype=np.float32,
        )
        rewards[-1] += float(game["reward"])
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in range(n - 1, -1, -1):
            next_value = 0.0 if t == n - 1 else float(values[t + 1])
            delta = float(rewards[t]) + gamma * next_value - float(values[t])
            gae = delta + gamma * lam * gae
            advantages[t] = gae
        returns = advantages + values
        all_adv.append(advantages)
        for r, adv, ret in zip(recs, advantages, returns):
            packed.append((r, float(adv), float(ret)))

    if not packed:
        raise RuntimeError("no strategic decisions collected")

    adv_all = np.concatenate(all_adv)
    adv_mean = float(adv_all.mean())
    adv_std = float(adv_all.std()) + 1e-6

    hidden = np.empty((len(packed), hidden_dim), np.float32)
    clock = np.empty((len(packed), clock_dim), np.float32)
    economic = np.empty((len(packed), ECONOMIC_DIM), np.float32)
    opponent = np.empty((len(packed), OPPONENT_DIM), np.float32)
    harvest = np.empty((len(packed), HARVEST_ECON_DIM), np.float32)
    base = np.empty(len(packed), np.int64)
    route_action = np.empty(len(packed), np.int64)
    route_relevant = np.empty(len(packed), np.bool_)
    market_action = np.empty(len(packed), np.int64)
    market_relevant = np.empty(len(packed), np.bool_)
    horizon_action = np.empty(len(packed), np.int64)
    horizon_relevant = np.empty(len(packed), np.bool_)
    route_bits = np.empty(len(packed), np.int64)
    market_bits = np.empty(len(packed), np.int64)
    old_logp = np.empty(len(packed), np.float32)
    advantages = np.empty(len(packed), np.float32)
    returns = np.empty(len(packed), np.float32)

    for i, (r, adv, ret) in enumerate(packed):
        h = np.frombuffer(r["hidden_f16"], np.float16)
        c = np.frombuffer(r["clock_f16"], np.float16)
        if h.size != hidden_dim or c.size != clock_dim:
            raise ValueError("trajectory tensor width mismatch")
        hidden[i] = h
        clock[i] = c
        economic[i] = _record_economic(r)
        opponent[i] = _record_opponent(r)
        harvest[i] = _record_harvest(r)
        base[i] = r["base_route_class"]
        route_action[i] = r["route_action"]
        route_relevant[i] = bool(r.get("route_relevant", True))
        market_action[i] = r["market_action"]
        market_relevant[i] = bool(r.get("market_relevant", True))
        horizon_action[i] = r["horizon_action"]
        horizon_relevant[i] = bool(
            r.get(
                "horizon_relevant",
                int(r["route_action"]) != int(r["base_route_class"]),
            )
        )
        route_bits[i] = r["route_mask_bits"]
        market_bits[i] = r["market_mask_bits"]
        old_logp[i] = r["old_logp"]
        advantages[i] = (adv - adv_mean) / adv_std
        returns[i] = ret

    def t(x, dtype=None):
        return torch.as_tensor(x, dtype=dtype, device=device)

    return {
        "hidden": t(hidden, torch.float32),
        "clock": t(clock, torch.float32),
        "economic": t(economic, torch.float32),
        "opponent": t(opponent, torch.float32),
        "harvest_economic": t(harvest, torch.float32),
        "base": t(base, torch.long),
        "route_action": t(route_action, torch.long),
        "route_relevant": t(route_relevant, torch.bool),
        "market_action": t(market_action, torch.long),
        "market_relevant": t(market_relevant, torch.bool),
        "horizon_action": t(horizon_action, torch.long),
        "horizon_relevant": t(horizon_relevant, torch.bool),
        "route_bits": t(route_bits, torch.long),
        "market_bits": t(market_bits, torch.long),
        "old_logp": t(old_logp, torch.float32),
        "advantage": t(advantages, torch.float32),
        "returns": t(returns, torch.float32),
    }


def _trajectory_signature(game):
    context = _game_economic_context(game)
    if context is None:
        economic_signature = ()
    else:
        # Preserve distinct shop/demand/ROI regimes even when two games happen
        # to produce the same strategic action sequence.
        keep = list(range(17)) + list(range(35, 43))
        economic_signature = tuple(
            int(round(float(context[i]) * 8.0)) for i in keep
        )
    return (
        int(game.get("seat", 0)),
        economic_signature,
        tuple(
            (
                int(r["step"]),
                int(r["route_action"]),
                int(r["market_action"]),
                (
                    int(r["horizon_action"])
                    if int(r["route_action"]) != int(r["base_route_class"])
                    else 0
                ),
            )
            for r in game.get("records", [])
        ),
    )


def update_elite_memory(path: Path, fresh_rows, *, limit: int, iteration: int):
    old = []
    if path.exists():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            old = list(payload.get("games", []))
        except Exception:
            old = []

    candidates = list(old)
    for row in fresh_rows:
        if not row.get("records"):
            continue
        g = dict(row)
        g["iteration_found"] = int(iteration)
        candidates.append(g)

    # Keep only the strongest instance of the same strategic action sequence.
    dedup = {}
    for game in candidates:
        sig = _trajectory_signature(game)
        previous = dedup.get(sig)
        if previous is None or (
            int(game["win"]),
            int(game["margin"]),
            int(game["own_money"]),
        ) > (
            int(previous["win"]),
            int(previous["margin"]),
            int(previous["own_money"]),
        ):
            dedup[sig] = game

    quality_key = lambda g: (
        int(g["win"]),
        int(g["margin"]),
        int(g["own_money"]),
    )
    all_games = sorted(dedup.values(), key=quality_key, reverse=True)
    limit_n = max(1, int(limit))

    # V4.3 reserves a quarter of memory for trajectories that actually carry
    # shop/economic context. Otherwise a full legacy top-256 buffer can reject
    # every fresh context trajectory before the new strategy branch can learn.
    context_reserve = min(64, max(1, limit_n // 4))
    contextual = [
        g for g in all_games if _game_economic_context(g) is not None
    ][:context_reserve]
    games = list(contextual)
    kept_ids = {id(g) for g in games}
    for game in all_games:
        if len(games) >= limit_n:
            break
        if id(game) in kept_ids:
            continue
        games.append(game)
        kept_ids.add(id(game))
    games.sort(key=quality_key, reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "farmos_v51_winner_memory_v4_3_context",
            "games": games,
        },
        path,
    )
    return games


def elite_summary(games):
    if not games:
        return {
            "elite_games": 0,
            "elite_wins": 0,
            "elite_best_margin": -999999999,
            "elite_median_margin": -999999999,
        }
    margins = [int(g["margin"]) for g in games]
    return {
        "elite_games": len(games),
        "elite_wins": sum(int(g["win"]) for g in games),
        "elite_best_margin": max(margins),
        "elite_median_margin": float(statistics.median(margins)),
    }


def partition_elite_games(games):
    winners = [g for g in games if float(g["margin"]) > 0.0]
    near = [g for g in games if -2000.0 <= float(g["margin"]) <= 0.0]
    good = [g for g in games if -8000.0 <= float(g["margin"]) < -2000.0]
    other = [g for g in games if float(g["margin"]) < -8000.0]
    key = lambda g: (
        int(g["win"]),
        int(g["margin"]),
        int(g["own_money"]),
    )
    return {
        "winner": sorted(winners, key=key, reverse=True),
        "near": sorted(near, key=key, reverse=True),
        "good": sorted(good, key=key, reverse=True),
        "other": sorted(other, key=key, reverse=True),
    }


def winner_summary(games):
    groups = partition_elite_games(games)
    return {
        "winner_count": len(groups["winner"]),
        "near_win_count": len(groups["near"]),
        "good_count": len(groups["good"]),
        "other_count": len(groups["other"]),
        "context_game_count": sum(
            _game_economic_context(g) is not None for g in games
        ),
        "best_winner_margin": (
            int(groups["winner"][0]["margin"])
            if groups["winner"] else None
        ),
        "best_near_margin": (
            int(groups["near"][0]["margin"])
            if groups["near"] else None
        ),
    }


def winner_records_to_batch(
    games,
    hidden_dim,
    clock_dim,
    device,
    *,
    max_winners=32,
    max_near=48,
    max_good=48,
    include_other=False,
    winner_only=False,
    context_rows=None,
):
    groups = partition_elite_games(games)
    reference_contexts = []
    for row in context_rows or []:
        context = _game_economic_context(row)
        if context is not None:
            reference_contexts.append(context)

    def contextual_order(items):
        if not reference_contexts:
            return list(items)
        return sorted(
            items,
            key=lambda g: (
                _context_similarity(g, reference_contexts),
                int(g["margin"]),
                int(g["own_money"]),
            ),
            reverse=True,
        )

    winner_games = contextual_order(groups["winner"])
    near_games = contextual_order(groups["near"])
    good_games = contextual_order(groups["good"])
    other_games = contextual_order(groups["other"])

    selected = []
    if winner_only and winner_games:
        selected += [(g, 100.0, "winner") for g in winner_games[:max_winners]]
    else:
        # V4.3: near-wins are the main discovery curriculum. Winners remain the
        # safety anchor, but no longer dominate every update.
        selected += [(g, 18.0, "winner") for g in winner_games[:max_winners]]
        selected += [(g, 22.0, "near") for g in near_games[:max_near]]
        selected += [(g, 3.0, "good") for g in good_games[:max_good]]
        if include_other:
            selected += [(g, 0.5, "other") for g in other_games[:16]]
    if not selected:
        return None

    packed = []
    for rank, (game, class_weight, kind) in enumerate(selected):
        rank_weight = 1.0 / math.sqrt(1.0 + rank / 8.0)
        similarity = _context_similarity(game, reference_contexts)
        context_weight = 0.50 + 1.50 * similarity
        game_weight = float(class_weight * rank_weight * context_weight)
        value_target = terminal_reward(game["own_money"], game["v51_money"])
        for record in game["records"]:
            packed.append((record, game_weight, value_target, kind))

    hidden = np.empty((len(packed), hidden_dim), np.float32)
    clock = np.empty((len(packed), clock_dim), np.float32)
    economic = np.empty((len(packed), ECONOMIC_DIM), np.float32)
    opponent = np.empty((len(packed), OPPONENT_DIM), np.float32)
    harvest = np.empty((len(packed), HARVEST_ECON_DIM), np.float32)
    base = np.empty(len(packed), np.int64)
    route_action = np.empty(len(packed), np.int64)
    route_relevant = np.empty(len(packed), np.bool_)
    market_action = np.empty(len(packed), np.int64)
    market_relevant = np.empty(len(packed), np.bool_)
    horizon_action = np.empty(len(packed), np.int64)
    horizon_relevant = np.empty(len(packed), np.bool_)
    route_bits = np.empty(len(packed), np.int64)
    market_bits = np.empty(len(packed), np.int64)
    weight = np.empty(len(packed), np.float32)
    value_target = np.empty(len(packed), np.float32)
    winner_mask = np.empty(len(packed), np.bool_)

    for i, (r, w, v, kind) in enumerate(packed):
        h = np.frombuffer(r["hidden_f16"], np.float16)
        c = np.frombuffer(r["clock_f16"], np.float16)
        if h.size != hidden_dim or c.size != clock_dim:
            raise ValueError("winner trajectory tensor width mismatch")
        hidden[i] = h
        clock[i] = c
        economic[i] = _record_economic(r)
        opponent[i] = _record_opponent(r)
        harvest[i] = _record_harvest(r)
        base[i] = r["base_route_class"]
        route_action[i] = r["route_action"]
        route_relevant[i] = bool(r.get("route_relevant", True))
        market_action[i] = r["market_action"]
        # Old V4 memory has no semantic market relevance flag. Preserve
        # supervision only for exact winners; near/good legacy labels can be
        # no-op stochastic noise. Fresh V4.1 records carry the exact flag.
        market_relevant[i] = bool(
            r.get("market_relevant", kind == "winner")
        )
        if not market_relevant[i]:
            market_action[i] = 0  # canonical KEEP_ROUTE for no-op market labels
        horizon_action[i] = r["horizon_action"]
        horizon_relevant[i] = bool(
            r.get(
                "horizon_relevant",
                int(r["route_action"]) != int(r["base_route_class"]),
            )
        )
        route_bits[i] = r["route_mask_bits"]
        market_bits[i] = r["market_mask_bits"]
        weight[i] = w
        value_target[i] = v
        winner_mask[i] = kind == "winner"

    def t(x, dtype=None):
        return torch.as_tensor(x, dtype=dtype, device=device)

    return {
        "hidden": t(hidden, torch.float32),
        "clock": t(clock, torch.float32),
        "economic": t(economic, torch.float32),
        "opponent": t(opponent, torch.float32),
        "harvest_economic": t(harvest, torch.float32),
        "base": t(base, torch.long),
        "route_action": t(route_action, torch.long),
        "route_relevant": t(route_relevant, torch.bool),
        "market_action": t(market_action, torch.long),
        "market_relevant": t(market_relevant, torch.bool),
        "horizon_action": t(horizon_action, torch.long),
        "horizon_relevant": t(horizon_relevant, torch.bool),
        "route_bits": t(route_bits, torch.long),
        "market_bits": t(market_bits, torch.long),
        "weight": t(weight, torch.float32),
        "value_target": t(value_target, torch.float32),
        "winner_mask": t(winner_mask, torch.bool),
        "winner_rows": int(winner_mask.sum()),
        "rows": len(packed),
    }


def elite_records_to_batch(games, hidden_dim, clock_dim, device, max_games):
    selected = sorted(
        games,
        key=lambda g: (int(g["win"]), int(g["margin"]), int(g["own_money"])),
        reverse=True,
    )[: max(1, int(max_games))]
    packed = []
    for rank, game in enumerate(selected):
        margin = float(game["margin"])
        # Near-win and winning trajectories get much more replay pressure.
        quality = 1.0 + 5.0 / (1.0 + math.exp(-(margin + 8000.0) / 2500.0))
        if margin > -2000.0:
            quality += 3.0
        if margin > 0.0:
            quality += 10.0
        rank_weight = 1.0 / math.sqrt(1.0 + rank / 16.0)
        weight = float(quality * rank_weight)
        value_target = terminal_reward(game["own_money"], game["v51_money"])
        for record in game["records"]:
            packed.append((record, weight, value_target))
    if not packed:
        return None

    hidden = np.empty((len(packed), hidden_dim), np.float32)
    clock = np.empty((len(packed), clock_dim), np.float32)
    economic = np.empty((len(packed), ECONOMIC_DIM), np.float32)
    opponent = np.empty((len(packed), OPPONENT_DIM), np.float32)
    harvest = np.empty((len(packed), HARVEST_ECON_DIM), np.float32)
    base = np.empty(len(packed), np.int64)
    route_action = np.empty(len(packed), np.int64)
    route_relevant = np.empty(len(packed), np.bool_)
    market_action = np.empty(len(packed), np.int64)
    market_relevant = np.empty(len(packed), np.bool_)
    horizon_action = np.empty(len(packed), np.int64)
    horizon_relevant = np.empty(len(packed), np.bool_)
    route_bits = np.empty(len(packed), np.int64)
    market_bits = np.empty(len(packed), np.int64)
    weight = np.empty(len(packed), np.float32)
    value_target = np.empty(len(packed), np.float32)

    for i, (r, w, v) in enumerate(packed):
        h = np.frombuffer(r["hidden_f16"], np.float16)
        c = np.frombuffer(r["clock_f16"], np.float16)
        if h.size != hidden_dim or c.size != clock_dim:
            raise ValueError("elite trajectory tensor width mismatch")
        hidden[i] = h
        clock[i] = c
        economic[i] = _record_economic(r)
        opponent[i] = _record_opponent(r)
        harvest[i] = _record_harvest(r)
        base[i] = r["base_route_class"]
        route_action[i] = r["route_action"]
        route_relevant[i] = bool(r.get("route_relevant", True))
        market_action[i] = r["market_action"]
        market_relevant[i] = bool(r.get("market_relevant", False))
        if not market_relevant[i]:
            market_action[i] = 0  # canonical KEEP_ROUTE for no-op market labels
        horizon_action[i] = r["horizon_action"]
        horizon_relevant[i] = bool(
            r.get(
                "horizon_relevant",
                int(r["route_action"]) != int(r["base_route_class"]),
            )
        )
        route_bits[i] = r["route_mask_bits"]
        market_bits[i] = r["market_mask_bits"]
        weight[i] = w
        value_target[i] = v

    def t(x, dtype=None):
        return torch.as_tensor(x, dtype=dtype, device=device)

    return {
        "hidden": t(hidden, torch.float32),
        "clock": t(clock, torch.float32),
        "economic": t(economic, torch.float32),
        "opponent": t(opponent, torch.float32),
        "harvest_economic": t(harvest, torch.float32),
        "base": t(base, torch.long),
        "route_action": t(route_action, torch.long),
        "route_relevant": t(route_relevant, torch.bool),
        "market_action": t(market_action, torch.long),
        "market_relevant": t(market_relevant, torch.bool),
        "horizon_action": t(horizon_action, torch.long),
        "horizon_relevant": t(horizon_relevant, torch.bool),
        "route_bits": t(route_bits, torch.long),
        "market_bits": t(market_bits, torch.long),
        "weight": t(weight, torch.float32),
        "value_target": t(value_target, torch.float32),
    }


def elite_bc_update(
    actor,
    parent,
    elite_batch,
    optimizer,
    harvest_optimizer,
    route_count,
    market_count,
    *,
    residual_scale,
    base_keep_bias,
    epochs,
    minibatch,
):
    if elite_batch is None or int(epochs) <= 0:
        return {"elite_loss": 0.0, "elite_route_acc": 0.0, "elite_rows": 0}

    actor.train()
    n = int(elite_batch["hidden"].shape[0])
    stats = []
    for _ in range(int(epochs)):
        perm = torch.randperm(n, device=elite_batch["hidden"].device)
        for start in range(0, n, minibatch):
            idx = perm[start : start + minibatch]
            h = elite_batch["hidden"][idx]
            c = elite_batch["clock"][idx]
            base = elite_batch["base"][idx]
            route_target = elite_batch["route_action"][idx]
            market_target = elite_batch["market_action"][idx]
            out = actor(
                h,
                c,
                base,
                economic=elite_batch["economic"][idx],
                opponent=elite_batch["opponent"][idx],
                harvest_economic=elite_batch["harvest_economic"][idx],
                route_action=route_target,
                market_action=market_target,
            )
            with torch.no_grad():
                parent_route = parent.route_logits_from_hidden(h, c)

            route_mask = bitmask_to_mask(
                elite_batch["route_bits"][idx], route_count
            )
            market_mask = bitmask_to_mask(
                elite_batch["market_bits"][idx], market_count
            )
            keep_bias = torch.nn.functional.one_hot(
                base, route_count
            ).to(parent_route.dtype) * base_keep_bias
            route_logits = (
                parent_route
                + keep_bias
                + residual_scale * out["route_residual"]
            ).masked_fill(~route_mask, torch.finfo(parent_route.dtype).min)
            market_logits = out["market"].masked_fill(
                ~market_mask, torch.finfo(out["market"].dtype).min
            )
            horizon_logits = out["horizon"]

            w = elite_batch["weight"][idx]
            w = w / w.mean().clamp_min(1e-6)
            route_per = torch.nn.functional.cross_entropy(
                route_logits, route_target, reduction="none"
            )
            market_per = torch.nn.functional.cross_entropy(
                market_logits, market_target, reduction="none"
            )
            horizon_per = torch.nn.functional.cross_entropy(
                horizon_logits,
                elite_batch["horizon_action"][idx],
                reduction="none",
            )
            route_rel = elite_batch["route_relevant"][idx]
            route_w = w * route_rel.to(w.dtype)
            if bool(route_rel.any()):
                route_loss = (
                    route_per * route_w
                ).sum() / route_w.sum().clamp_min(1e-6)
            else:
                route_loss = route_per.sum() * 0.0
            market_rel = elite_batch["market_relevant"][idx]
            market_w = w * market_rel.to(w.dtype)
            if bool(market_rel.any()):
                market_loss = (
                    market_per * market_w
                ).sum() / market_w.sum().clamp_min(1e-6)
            else:
                market_loss = market_per.sum() * 0.0
            horizon_rel = elite_batch["horizon_relevant"][idx]
            horizon_w = w * horizon_rel.to(w.dtype)
            if bool(horizon_rel.any()):
                horizon_loss = (
                    horizon_per * horizon_w
                ).sum() / horizon_w.sum().clamp_min(1e-6)
            else:
                horizon_loss = horizon_per.sum() * 0.0
            value_loss = torch.nn.functional.smooth_l1_loss(
                out["value"], elite_batch["value_target"][idx]
            )
            loss = (
                route_loss
                + 0.60 * market_loss
                + 0.35 * horizon_loss
                + 0.05 * value_loss
            )
            optimizer.zero_grad(set_to_none=True)
            harvest_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.7)
            optimizer.step()
            harvest_optimizer.step()
            with torch.no_grad():
                if bool(route_rel.any()):
                    acc = float(
                        (
                            route_logits.argmax(-1)[route_rel]
                            == route_target[route_rel]
                        ).float().mean()
                    )
                else:
                    acc = float("nan")
            stats.append((float(loss.detach()), acc))
    actor.eval()
    route_accs = [x[1] for x in stats if math.isfinite(x[1])]
    return {
        "elite_loss": statistics.fmean(x[0] for x in stats),
        "elite_route_acc": (
            statistics.fmean(route_accs) if route_accs else 0.0
        ),
        "elite_rows": n,
    }


def winner_bc_update(
    actor,
    parent,
    winner_batch,
    optimizer,
    harvest_optimizer,
    route_count,
    market_count,
    *,
    residual_scale,
    base_keep_bias,
    epochs,
    minibatch,
):
    if winner_batch is None or int(epochs) <= 0:
        return {
            "winner_loss": 0.0,
            "route_acc": 0.0,
            "market_acc": 0.0,
            "horizon_acc": 0.0,
            "winner_route_prob": 0.0,
            "winner_market_prob": 0.0,
            "winner_horizon_prob": 0.0,
            "rows": 0,
            "winner_rows": 0,
            "market_rows": 0,
            "horizon_rows": 0,
        }

    actor.train()
    n = int(winner_batch["hidden"].shape[0])
    stats = []
    for _ in range(int(epochs)):
        perm = torch.randperm(n, device=winner_batch["hidden"].device)
        for start in range(0, n, minibatch):
            idx = perm[start : start + minibatch]
            h = winner_batch["hidden"][idx]
            c = winner_batch["clock"][idx]
            base = winner_batch["base"][idx]
            route_target = winner_batch["route_action"][idx]
            route_rel = winner_batch["route_relevant"][idx]
            market_target = winner_batch["market_action"][idx]
            market_rel = winner_batch["market_relevant"][idx]
            horizon_target = winner_batch["horizon_action"][idx]
            horizon_rel = winner_batch["horizon_relevant"][idx]

            # Teacher-force route -> market -> horizon for distillation. This
            # matches rollout order and removes the old independent-head mismatch.
            out = actor(
                h,
                c,
                base,
                economic=winner_batch["economic"][idx],
                opponent=winner_batch["opponent"][idx],
                harvest_economic=winner_batch["harvest_economic"][idx],
                route_action=route_target,
                market_action=market_target,
            )
            with torch.no_grad():
                parent_route = parent.route_logits_from_hidden(h, c)

            route_mask = bitmask_to_mask(
                winner_batch["route_bits"][idx], route_count
            )
            market_mask = bitmask_to_mask(
                winner_batch["market_bits"][idx], market_count
            )
            keep_bias = torch.nn.functional.one_hot(
                base, route_count
            ).to(parent_route.dtype) * base_keep_bias
            route_logits = (
                parent_route
                + keep_bias
                + residual_scale * out["route_residual"]
            ).masked_fill(~route_mask, torch.finfo(parent_route.dtype).min)
            market_logits = out["market"].masked_fill(
                ~market_mask, torch.finfo(out["market"].dtype).min
            )
            horizon_logits = out["horizon"]

            w = winner_batch["weight"][idx]
            w = w / w.mean().clamp_min(1e-6)
            route_per = torch.nn.functional.cross_entropy(
                route_logits, route_target, reduction="none"
            )
            market_per = torch.nn.functional.cross_entropy(
                market_logits, market_target, reduction="none"
            )
            horizon_per = torch.nn.functional.cross_entropy(
                horizon_logits, horizon_target, reduction="none"
            )
            route_w = w * route_rel.to(w.dtype)
            if bool(route_rel.any()):
                route_loss = (
                    route_per * route_w
                ).sum() / route_w.sum().clamp_min(1e-6)
            else:
                route_loss = route_per.sum() * 0.0
            market_w = w * market_rel.to(w.dtype)
            if bool(market_rel.any()):
                market_loss = (
                    market_per * market_w
                ).sum() / market_w.sum().clamp_min(1e-6)
            else:
                market_loss = market_per.sum() * 0.0

            horizon_w = w * horizon_rel.to(w.dtype)
            if bool(horizon_rel.any()):
                horizon_loss = (
                    horizon_per * horizon_w
                ).sum() / horizon_w.sum().clamp_min(1e-6)
            else:
                horizon_loss = horizon_per.sum() * 0.0

            value_loss = torch.nn.functional.smooth_l1_loss(
                out["value"], winner_batch["value_target"][idx]
            )
            loss = (
                0.85 * route_loss
                + 1.40 * market_loss
                + 0.90 * horizon_loss
                + 0.03 * value_loss
            )
            optimizer.zero_grad(set_to_none=True)
            harvest_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            optimizer.step()
            harvest_optimizer.step()

            with torch.no_grad():
                route_prob = torch.softmax(route_logits, -1).gather(
                    1, route_target.unsqueeze(1)
                ).squeeze(1)
                market_prob = torch.softmax(market_logits, -1).gather(
                    1, market_target.unsqueeze(1)
                ).squeeze(1)
                horizon_prob = torch.softmax(horizon_logits, -1).gather(
                    1, horizon_target.unsqueeze(1)
                ).squeeze(1)
                if bool(route_rel.any()):
                    route_acc = float(
                        (
                            route_logits.argmax(-1)[route_rel]
                            == route_target[route_rel]
                        ).float().mean()
                    )
                else:
                    route_acc = float("nan")
                if bool(market_rel.any()):
                    market_acc = float(
                        (
                            market_logits.argmax(-1)[market_rel]
                            == market_target[market_rel]
                        ).float().mean()
                    )
                else:
                    market_acc = float("nan")
                if bool(horizon_rel.any()):
                    horizon_acc = float(
                        (
                            horizon_logits.argmax(-1)[horizon_rel]
                            == horizon_target[horizon_rel]
                        ).float().mean()
                    )
                else:
                    horizon_acc = float("nan")

                wm = winner_batch["winner_mask"][idx]
                wm_route = wm & route_rel
                if bool(wm_route.any()):
                    wrp = float(route_prob[wm_route].mean())
                else:
                    wrp = float("nan")
                wm_market = wm & market_rel
                if bool(wm_market.any()):
                    wmp = float(market_prob[wm_market].mean())
                else:
                    wmp = float("nan")
                wh_mask = wm & horizon_rel
                if bool(wh_mask.any()):
                    whp = float(horizon_prob[wh_mask].mean())
                else:
                    whp = float("nan")

            stats.append(
                (
                    float(loss.detach()),
                    float(route_acc),
                    float(market_acc),
                    horizon_acc,
                    wrp,
                    wmp,
                    whp,
                    int(horizon_rel.sum().item()),
                )
            )
    actor.eval()

    def mean_finite(pos):
        vals = [x[pos] for x in stats if math.isfinite(x[pos])]
        return statistics.fmean(vals) if vals else 0.0

    return {
        "winner_loss": statistics.fmean(x[0] for x in stats),
        "route_acc": mean_finite(1),
        "market_acc": mean_finite(2),
        "horizon_acc": mean_finite(3),
        "winner_route_prob": mean_finite(4),
        "winner_market_prob": mean_finite(5),
        "winner_horizon_prob": mean_finite(6),
        "rows": n,
        "winner_rows": int(winner_batch["winner_rows"]),
        "market_rows": int(
            winner_batch["market_relevant"].sum().item()
        ),
        "horizon_rows": int(
            winner_batch["horizon_relevant"].sum().item()
        ),
    }


def split_skill_teacher_rows(rows, *, holdout_mod=5):
    """Deterministic game-level split so held-out games never enter BC replay."""
    train_rows, val_rows = [], []
    mod = max(2, int(holdout_mod))
    for game in rows:
        seed = int(game.get("seed", 0) or 0)
        seat = int(game.get("seat", 0) or 0)
        bucket = (seed * 1009 + seat * 97) % mod
        (val_rows if bucket == 0 else train_rows).append(game)
    return train_rows, val_rows


def skill_teacher_class_counts(rows):
    counts = [0 for _ in MICRO_TASKS]
    for game in rows:
        for record in (game.get("skill_teacher_records") or []):
            idx = int(record.get("teacher_task_action", -1))
            if 0 <= idx < len(MICRO_TASKS):
                counts[idx] += 1
    return counts


def update_skill_teacher_memory(path, rows, *, per_class_limit=2048, iteration=0):
    """Persistent class-capped replay for V4.5 shadow skill imitation."""
    previous = []
    if path.exists():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            previous = list(payload.get("records", []) or [])
        except Exception:
            previous = []
    incoming = [
        r
        for game in rows
        for r in (game.get("skill_teacher_records") or [])
    ]
    buckets = {i: [] for i in range(len(MICRO_TASKS))}
    for record in previous + incoming:
        idx = int(record.get("teacher_task_action", -1))
        if 0 <= idx < len(MICRO_TASKS):
            buckets[idx].append(record)
    rng = random.Random(45_000_000 + int(iteration))
    kept = []
    limit = max(32, int(per_class_limit))
    for idx in range(len(MICRO_TASKS)):
        bucket = buckets[idx]
        if len(bucket) > limit:
            recent_n = limit // 2
            recent = bucket[-recent_n:]
            older = bucket[:-recent_n]
            sample_n = limit - len(recent)
            sampled = rng.sample(older, sample_n) if len(older) > sample_n else older
            bucket = sampled + recent
        kept.extend(bucket)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "farmos_v45_skill_teacher_memory_v1",
            "iteration": int(iteration),
            "per_class_limit": int(limit),
            "records": kept,
        },
        path,
    )
    counts = [0 for _ in MICRO_TASKS]
    for record in kept:
        counts[int(record["teacher_task_action"])] += 1
    return kept, counts


def skill_teacher_records_to_batch(
    rows, hidden_dim, clock_dim, device, *, natural_counts=None, prior_power=0.50
):
    records = [
        r
        for game in rows
        for r in (game.get("skill_teacher_records") or [])
    ]
    if not records:
        return None
    n = len(records)
    hidden = np.empty((n, hidden_dim), np.float32)
    clock = np.empty((n, clock_dim), np.float32)
    local = np.empty((n, MICRO_LOCAL_DIM), np.float32)
    plant_context = np.empty((n, PLANT_CONTEXT_DIM), np.float32)
    economic = np.empty((n, ECONOMIC_DIM), np.float32)
    opponent = np.empty((n, OPPONENT_DIM), np.float32)
    target = np.empty(n, np.int64)
    task_bits = np.empty(n, np.int64)
    weight = np.empty(n, np.float32)
    for i, record in enumerate(records):
        h = np.frombuffer(record["hidden_f16"], np.float16)
        c = np.frombuffer(record["clock_f16"], np.float16)
        l = np.frombuffer(record["local_f16"], np.float16)
        if h.size != hidden_dim or c.size != clock_dim or l.size != MICRO_LOCAL_DIM:
            raise ValueError("skill teacher tensor width mismatch")
        hidden[i] = h
        clock[i] = c
        local[i] = l
        plant_context[i] = _record_plant_context(record)
        economic[i] = _record_economic(record)
        opponent[i] = _record_opponent(record)
        target[i] = int(record["teacher_task_action"])
        task_bits[i] = int(
            record.get("task_mask_bits", (1 << len(MICRO_TASKS)) - 1)
        )
        weight[i] = 1.0

    # Replay remains class-capped so rare structural skills are never lost,
    # but a fully uniform 2048/class replay destroys the real task prior and
    # causes severe rare-class false positives (notably FERTILIZE). Reapply a
    # softened natural prior from fresh TRAIN games while retaining the replay
    # balancing term. alpha=0.5 is deliberately conservative: sqrt prior.
    class_counts = np.bincount(
        target, minlength=len(MICRO_TASKS)
    ).astype(np.float32)
    nonkeep_counts = class_counts[1:][class_counts[1:] > 0]
    reference = float(np.median(nonkeep_counts)) if nonkeep_counts.size else 1.0
    natural = None
    natural_ref = 1.0
    if natural_counts is not None and len(natural_counts) == len(MICRO_TASKS):
        natural = np.asarray(natural_counts, dtype=np.float32)
        natural_nonkeep = natural[1:][natural[1:] > 0]
        if natural_nonkeep.size:
            natural_ref = float(np.median(natural_nonkeep))
    class_weight = np.ones(len(MICRO_TASKS), dtype=np.float32)
    class_weight[0] = 0.05
    alpha = max(0.0, min(1.0, float(prior_power)))
    for task_idx in range(1, len(MICRO_TASKS)):
        count = float(class_counts[task_idx])
        if count <= 0:
            continue
        replay_balance = math.sqrt(reference / max(1.0, count))
        prior = 1.0
        if natural is not None and float(natural[task_idx]) > 0:
            prior = (float(natural[task_idx]) / max(1.0, natural_ref)) ** alpha
        combined = replay_balance * prior
        class_weight[task_idx] = 2.0 * float(
            max(0.35, min(3.0, combined))
        )
    weight[:] = class_weight[target]

    def t(x, dtype=None):
        return torch.as_tensor(x, dtype=dtype, device=device)

    return {
        "hidden": t(hidden, torch.float32),
        "clock": t(clock, torch.float32),
        "local": t(local, torch.float32),
        "plant_context": t(plant_context, torch.float32),
        "economic": t(economic, torch.float32),
        "opponent": t(opponent, torch.float32),
        "target": t(target, torch.long),
        "task_bits": t(task_bits, torch.long),
        "weight": t(weight, torch.float32),
    }


def skill_bc_update(
    actor,
    batch,
    optimizer,
    *,
    temperature,
    epochs=2,
    minibatch=512,
    keep_penalty=1.25,
    update=True,
):
    if batch is None or batch["hidden"].shape[0] == 0:
        return {
            "rows": 0,
            "loss": 0.0,
            "acc": 0.0,
            "nonkeep_acc": 0.0,
            "nonkeep_rows": 0,
            "target_counts": [0 for _ in MICRO_TASKS],
            "per_task_acc": [0.0 for _ in MICRO_TASKS],
            "per_task_precision": [0.0 for _ in MICRO_TASKS],
            "per_task_recall": [0.0 for _ in MICRO_TASKS],
            "per_task_f1": [0.0 for _ in MICRO_TASKS],
            "core_min_acc": 0.0,
            "core_min_f1": 0.0,
            "confusion": [[0 for _ in MICRO_TASKS] for _ in MICRO_TASKS],
        }
    actor.train() if bool(update) else actor.eval()
    n = int(batch["hidden"].shape[0])
    stats = []
    micro_params = (
        list(actor.micro_trunk.parameters())
        + list(actor.micro_plant_context.parameters())
        + list(actor.micro_task.parameters())
        + list(actor.micro_value.parameters())
    )
    for _ in range(max(1, int(epochs)) if bool(update) else 0):
        perm = torch.randperm(n, device=batch["hidden"].device)
        for start in range(0, n, max(1, int(minibatch))):
            idx = perm[start : start + minibatch]
            out = actor.micro_forward(
                batch["hidden"][idx],
                batch["clock"][idx],
                batch["local"][idx],
                economic=batch["economic"][idx],
                opponent=batch["opponent"][idx],
                plant_context=batch["plant_context"][idx],
            )
            logits = out["task"] / temperature
            mask = bitmask_to_mask(batch["task_bits"][idx], len(MICRO_TASKS))
            nonkeep_valid = mask[:, 1:].any(dim=-1)
            if bool(nonkeep_valid.any()):
                logits = logits.clone()
                logits[nonkeep_valid, 0] -= float(keep_penalty)
            logits = logits.masked_fill(
                ~mask, torch.finfo(logits.dtype).min
            )
            target = batch["target"][idx]
            per_row = torch.nn.functional.cross_entropy(
                logits, target, reduction="none"
            )
            weights = batch["weight"][idx]
            loss = (per_row * weights).sum() / weights.sum().clamp_min(1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(micro_params, 0.5)
            optimizer.step()
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = float((pred == target).float().mean())
                nk = target != 0
                nk_acc = (
                    float((pred[nk] == target[nk]).float().mean())
                    if bool(nk.any())
                    else float("nan")
                )
                stats.append((float(loss.detach()), acc, nk_acc))
    actor.eval()
    target_full = batch["target"]
    target_counts = torch.bincount(
        target_full, minlength=len(MICRO_TASKS)
    ).detach().cpu().tolist()
    pred_chunks = []
    with torch.no_grad():
        eval_batch = max(1024, int(minibatch))
        for start in range(0, n, eval_batch):
            idx = slice(start, min(n, start + eval_batch))
            out = actor.micro_forward(
                batch["hidden"][idx],
                batch["clock"][idx],
                batch["local"][idx],
                economic=batch["economic"][idx],
                opponent=batch["opponent"][idx],
                plant_context=batch["plant_context"][idx],
            )
            logits = out["task"] / temperature
            mask = bitmask_to_mask(
                batch["task_bits"][idx], len(MICRO_TASKS)
            )
            nonkeep_valid = mask[:, 1:].any(dim=-1)
            if bool(nonkeep_valid.any()):
                logits = logits.clone()
                logits[nonkeep_valid, 0] -= float(keep_penalty)
            logits = logits.masked_fill(
                ~mask, torch.finfo(logits.dtype).min
            )
            pred_chunks.append(logits.argmax(dim=-1))
    pred_full = torch.cat(pred_chunks, dim=0)
    final_acc = float((pred_full == target_full).float().mean().item())
    nonkeep_mask = target_full != 0
    nonkeep_rows = int(nonkeep_mask.sum().item())
    final_nonkeep_acc = (
        float((pred_full[nonkeep_mask] == target_full[nonkeep_mask]).float().mean().item())
        if nonkeep_rows > 0 else 0.0
    )
    per_task_acc = []
    per_task_precision = []
    per_task_recall = []
    per_task_f1 = []
    for task_idx in range(len(MICRO_TASKS)):
        true_mask = target_full == task_idx
        pred_mask = pred_full == task_idx
        tp = int((true_mask & pred_mask).sum().item())
        fp = int((~true_mask & pred_mask).sum().item())
        fn = int((true_mask & ~pred_mask).sum().item())
        if bool(true_mask.any()):
            recall = tp / max(1, tp + fn)
            per_task_acc.append(float(recall))
            per_task_recall.append(float(recall))
        else:
            per_task_acc.append(float("nan"))
            per_task_recall.append(float("nan"))
        if bool(pred_mask.any()):
            precision = tp / max(1, tp + fp)
            per_task_precision.append(float(precision))
        else:
            per_task_precision.append(float("nan"))
            precision = float("nan")
        recall_v = per_task_recall[-1]
        if math.isfinite(precision) and math.isfinite(recall_v) and precision + recall_v > 0:
            per_task_f1.append(float(2.0 * precision * recall_v / (precision + recall_v)))
        elif math.isfinite(precision) and math.isfinite(recall_v):
            per_task_f1.append(0.0)
        else:
            per_task_f1.append(float("nan"))
    # Promotion is blocked by the weakest operational skill, not just by
    # aggregate accuracy. This prevents common WATER/CARE examples from
    # hiding failures in fertilizer, storage, weeding, planting or deployment.
    core_names = tuple(MICRO_TASKS[1:])
    core_acc = []
    for name in core_names:
        task_idx = MICRO_TASKS.index(name)
        if target_counts[task_idx] >= 32 and math.isfinite(per_task_acc[task_idx]):
            core_acc.append(per_task_acc[task_idx])
    core_min_acc = min(core_acc) if core_acc else 0.0
    core_f1 = []
    for name in core_names:
        task_idx = MICRO_TASKS.index(name)
        if target_counts[task_idx] >= 32 and math.isfinite(per_task_f1[task_idx]):
            core_f1.append(per_task_f1[task_idx])
    core_min_f1 = min(core_f1) if core_f1 else 0.0
    k = len(MICRO_TASKS)
    confusion = torch.bincount(
        target_full * k + pred_full, minlength=k * k
    ).reshape(k, k).detach().cpu().tolist()
    return {
        "rows": n,
        "loss": (statistics.fmean(x[0] for x in stats) if stats else 0.0),
        "acc": final_acc,
        "nonkeep_acc": final_nonkeep_acc,
        "nonkeep_rows": nonkeep_rows,
        "target_counts": target_counts,
        "per_task_acc": per_task_acc,
        "per_task_precision": per_task_precision,
        "per_task_recall": per_task_recall,
        "per_task_f1": per_task_f1,
        "core_min_acc": core_min_acc,
        "core_min_f1": core_min_f1,
        "confusion": confusion,
    }


def micro_records_to_training(
    rows, hidden_dim, clock_dim, device, gamma=0.97, lam=0.90
):
    packed = []
    all_adv = []
    for game in rows:
        groups = {}
        for record in game.get("micro_records", []):
            groups.setdefault(int(record.get("hand_idx", 0)), []).append(record)
        for hand_rows in groups.values():
            hand_rows.sort(key=lambda r: int(r.get("step", 0)))
            if not hand_rows:
                continue
            values = np.asarray(
                [float(r.get("old_value", 0.0)) for r in hand_rows],
                dtype=np.float32,
            )
            rewards = np.asarray(
                [float(r.get("micro_reward", 0.0)) for r in hand_rows],
                dtype=np.float32,
            )
            advantages = np.zeros(len(hand_rows), dtype=np.float32)
            gae = 0.0
            for t in range(len(hand_rows) - 1, -1, -1):
                next_value = 0.0 if t == len(hand_rows) - 1 else float(values[t + 1])
                delta = float(rewards[t]) + gamma * next_value - float(values[t])
                gae = delta + gamma * lam * gae
                advantages[t] = gae
            returns = advantages + values
            all_adv.append(advantages)
            packed.extend(
                (record, float(adv), float(ret))
                for record, adv, ret in zip(hand_rows, advantages, returns)
            )
    if not packed:
        return None
    adv_all = np.concatenate(all_adv)
    adv_mean = float(adv_all.mean())
    adv_std = float(adv_all.std()) + 1e-6
    hidden = np.empty((len(packed), hidden_dim), np.float32)
    clock = np.empty((len(packed), clock_dim), np.float32)
    local = np.empty((len(packed), MICRO_LOCAL_DIM), np.float32)
    plant_context = np.empty((len(packed), PLANT_CONTEXT_DIM), np.float32)
    economic = np.empty((len(packed), ECONOMIC_DIM), np.float32)
    opponent = np.empty((len(packed), OPPONENT_DIM), np.float32)
    task = np.empty(len(packed), np.int64)
    task_bits = np.empty(len(packed), np.int64)
    skill_mode = np.empty(len(packed), np.bool_)
    old_logp = np.empty(len(packed), np.float32)
    advantage = np.empty(len(packed), np.float32)
    returns = np.empty(len(packed), np.float32)
    for i, (record, adv, ret) in enumerate(packed):
        h = np.frombuffer(record["hidden_f16"], np.float16)
        c = np.frombuffer(record["clock_f16"], np.float16)
        l = np.frombuffer(record["local_f16"], np.float16)
        if h.size != hidden_dim or c.size != clock_dim or l.size != MICRO_LOCAL_DIM:
            raise ValueError("micro trajectory tensor width mismatch")
        hidden[i] = h
        clock[i] = c
        local[i] = l
        plant_context[i] = _record_plant_context(record)
        economic[i] = _record_economic(record)
        opponent[i] = _record_opponent(record)
        task[i] = int(record["task_action"])
        task_bits[i] = int(
            record.get("task_mask_bits", (1 << len(MICRO_TASKS)) - 1)
        )
        skill_mode[i] = bool(record.get("skill_mode", False))
        old_logp[i] = float(record["old_logp"])
        advantage[i] = (adv - adv_mean) / adv_std
        returns[i] = ret
    def t(x, dtype=None):
        return torch.as_tensor(x, dtype=dtype, device=device)
    return {
        "hidden": t(hidden, torch.float32),
        "clock": t(clock, torch.float32),
        "local": t(local, torch.float32),
        "plant_context": t(plant_context, torch.float32),
        "economic": t(economic, torch.float32),
        "opponent": t(opponent, torch.float32),
        "task": t(task, torch.long),
        "task_bits": t(task_bits, torch.long),
        "skill_mode": t(skill_mode, torch.bool),
        "old_logp": t(old_logp, torch.float32),
        "advantage": t(advantage, torch.float32),
        "returns": t(returns, torch.float32),
    }


def ppo_update(
    actor,
    parent,
    batch,
    optimizer,
    harvest_optimizer,
    route_count,
    market_count,
    *,
    residual_scale,
    base_keep_bias,
    temperature,
    epochs,
    minibatch,
    clip_ratio,
    entropy_coef,
    value_coef,
    target_kl,
    update_macro=True,
):
    actor.train()
    n = batch["hidden"].shape[0]
    stats = []
    for epoch in range(epochs):
        perm = torch.randperm(n, device=batch["hidden"].device)
        for start in range(0, n, minibatch):
            idx = perm[start : start + minibatch]
            h = batch["hidden"][idx]
            c = batch["clock"][idx]
            base = batch["base"][idx]
            route_target = batch["route_action"][idx]
            market_target = batch["market_action"][idx]
            market_rel = batch["market_relevant"][idx]
            horizon_target = batch["horizon_action"][idx]
            horizon_rel = batch["horizon_relevant"][idx]
            out = actor(
                h,
                c,
                base,
                economic=batch["economic"][idx],
                opponent=batch["opponent"][idx],
                harvest_economic=batch["harvest_economic"][idx],
                route_action=route_target,
                market_action=market_target,
            )
            with torch.no_grad():
                parent_route = parent.route_logits_from_hidden(h, c)
            route_mask = bitmask_to_mask(batch["route_bits"][idx], route_count)
            market_mask = bitmask_to_mask(batch["market_bits"][idx], market_count)

            keep_bias = torch.nn.functional.one_hot(
                base, route_count
            ).to(parent_route.dtype) * base_keep_bias
            route_logits = (
                parent_route
                + keep_bias
                + residual_scale * out["route_residual"]
            ) / temperature
            route_logits = route_logits.masked_fill(
                ~route_mask, torch.finfo(route_logits.dtype).min
            )
            market_logits = (out["market"] / temperature).masked_fill(
                ~market_mask, torch.finfo(out["market"].dtype).min
            )
            horizon_logits = out["horizon"] / temperature

            route_dist = Categorical(logits=route_logits)
            market_dist = Categorical(logits=market_logits)
            horizon_dist = Categorical(logits=horizon_logits)
            route_gate = batch["route_relevant"][idx].to(route_logits.dtype)
            market_gate = market_rel.to(market_logits.dtype)
            horizon_gate = horizon_rel.to(horizon_logits.dtype)
            new_logp = (
                route_gate * route_dist.log_prob(route_target)
                + market_gate * market_dist.log_prob(market_target)
                + horizon_gate * horizon_dist.log_prob(horizon_target)
            )
            entropy = (
                route_gate * route_dist.entropy()
                + market_gate * market_dist.entropy()
                + horizon_gate * horizon_dist.entropy()
            ).mean()

            ratio = torch.exp(new_logp - batch["old_logp"][idx])
            adv = batch["advantage"][idx]
            s1 = ratio * adv
            s2 = torch.clamp(
                ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
            ) * adv
            policy_loss = -torch.minimum(s1, s2).mean()
            value_loss = torch.nn.functional.smooth_l1_loss(
                out["value"], batch["returns"][idx]
            )
            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            harvest_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if bool(update_macro):
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.7)
                optimizer.step()
            else:
                torch.nn.utils.clip_grad_norm_(
                    actor.harvest_router.parameters(), 0.35
                )
            harvest_optimizer.step()

            with torch.no_grad():
                approx_kl = (batch["old_logp"][idx] - new_logp).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > clip_ratio
                ).float().mean()
            stats.append(
                (
                    float(policy_loss.detach()),
                    float(value_loss.detach()),
                    float(entropy.detach()),
                    float(approx_kl.detach()),
                    float(clip_fraction.detach()),
                )
            )
        # Stop PPO earlier when the policy moves outside the trust region.
        # The previous 2.0x threshold allowed repeated KL spikes in live training.
        if stats and abs(stats[-1][3]) > target_kl * 1.5:
            break

    actor.eval()
    return {
        "policy_loss": statistics.fmean(x[0] for x in stats),
        "value_loss": statistics.fmean(x[1] for x in stats),
        "entropy": statistics.fmean(x[2] for x in stats),
        "approx_kl": statistics.fmean(x[3] for x in stats),
        "clip_fraction": statistics.fmean(x[4] for x in stats),
        "ppo_minibatches": len(stats),
    }


def micro_ppo_update(
    actor,
    batch,
    optimizer,
    *,
    temperature,
    epochs=2,
    minibatch=512,
    clip_ratio=0.15,
    entropy_coef=0.02,
    value_coef=0.20,
    target_kl=0.02,
    skill_keep_penalty=4.50,
    update=True,
):
    if batch is None or batch["hidden"].shape[0] == 0:
        return {
            "rows": 0, "policy_loss": 0.0, "value_loss": 0.0,
            "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0,
            "task_counts": [0] * len(MICRO_TASKS),
        }
    actor.train() if bool(update) else actor.eval()
    n = int(batch["hidden"].shape[0])
    stats = []
    micro_params = (
        list(actor.micro_trunk.parameters())
        + list(actor.micro_plant_context.parameters())
        + list(actor.micro_task.parameters())
        + list(actor.micro_value.parameters())
    )
    for _ in range(max(1, int(epochs)) if bool(update) else 0):
        perm = torch.randperm(n, device=batch["hidden"].device)
        for start in range(0, n, max(1, int(minibatch))):
            idx = perm[start : start + minibatch]
            out = actor.micro_forward(
                batch["hidden"][idx],
                batch["clock"][idx],
                batch["local"][idx],
                economic=batch["economic"][idx],
                opponent=batch["opponent"][idx],
                plant_context=batch["plant_context"][idx],
            )
            logits = out["task"] / temperature
            task_mask = bitmask_to_mask(
                batch["task_bits"][idx], len(MICRO_TASKS)
            )
            skill_rows = batch["skill_mode"][idx]
            nonkeep_valid = task_mask[:, 1:].any(dim=-1)
            penalize_keep = skill_rows & nonkeep_valid
            if bool(penalize_keep.any()):
                logits = logits.clone()
                logits[penalize_keep, 0] -= float(skill_keep_penalty)
            logits = logits.masked_fill(
                ~task_mask, torch.finfo(logits.dtype).min
            )
            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(batch["task"][idx])
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_logp - batch["old_logp"][idx])
            adv = batch["advantage"][idx]
            s1 = ratio * adv
            s2 = torch.clamp(
                ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
            ) * adv
            policy_loss = -torch.minimum(s1, s2).mean()
            value_loss = torch.nn.functional.smooth_l1_loss(
                out["value"], batch["returns"][idx]
            )
            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(micro_params, 0.5)
            optimizer.step()
            with torch.no_grad():
                approx_kl = (batch["old_logp"][idx] - new_logp).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > clip_ratio
                ).float().mean()
            stats.append((
                float(policy_loss.detach()),
                float(value_loss.detach()),
                float(entropy.detach()),
                float(approx_kl.detach()),
                float(clip_fraction.detach()),
            ))
        # Stop PPO earlier when the policy moves outside the trust region.
        # The previous 2.0x threshold allowed repeated KL spikes in live training.
        if stats and abs(stats[-1][3]) > target_kl * 1.5:
            break
    actor.eval()
    task_counts = torch.bincount(
        batch["task"].detach().cpu(), minlength=len(MICRO_TASKS)
    ).tolist()
    return {
        "rows": n,
        "policy_loss": (statistics.fmean(x[0] for x in stats) if stats else 0.0),
        "value_loss": (statistics.fmean(x[1] for x in stats) if stats else 0.0),
        "entropy": (statistics.fmean(x[2] for x in stats) if stats else 0.0),
        "approx_kl": (statistics.fmean(x[3] for x in stats) if stats else 0.0),
        "clip_fraction": (statistics.fmean(x[4] for x in stats) if stats else 0.0),
        "task_counts": task_counts,
    }


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_state(
    path,
    actor,
    optimizer,
    iteration,
    best_margin,
    best_win_rate,
    config,
    extra=None,
):
    payload = {
        "schema": "farmos_v51_winner_amplification_state_v4_5_skill_manager",
        "actor_state": {
            k: v.detach().cpu() for k, v in actor.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "iteration": int(iteration),
        "best_margin": float(best_margin),
        "best_win_rate": float(best_win_rate),
        "config": dict(config),
    }
    if extra:
        payload.update(dict(extra))
    torch.save(payload, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parent",
        default=str(ROOT / "assets" / "parent_promoted_v2.pt"),
    )
    ap.add_argument(
        "--opponent",
        default=str(ROOT / "assets" / "v51_main.py"),
    )
    ap.add_argument("--output-dir", default=str(ROOT / "output"))
    ap.add_argument("--workers", type=int, default=max(2, min(32, (os.cpu_count() or 8) // 2)))
    ap.add_argument("--train-seeds-per-iter", type=int, default=max(2, min(32, (os.cpu_count() or 8) // 2)))
    ap.add_argument("--worker-recycle-tasks", type=int, default=64)
    ap.add_argument("--seed-base", type=int, default=23200000)
    ap.add_argument("--eval-seeds", default="22990000:4")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--decision-every", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=1.30)
    ap.add_argument("--min-temperature", type=float, default=1.00)
    ap.add_argument("--residual-scale", type=float, default=1.0)
    ap.add_argument("--base-keep-bias", type=float, default=2.3)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--harvest-learning-rate", type=float, default=2.5e-5)
    ap.add_argument("--harvest-only-warmup-iters", type=int, default=0)
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--elite-limit", type=int, default=256)
    ap.add_argument("--elite-train-games", type=int, default=96)
    ap.add_argument("--elite-epochs", type=int, default=2)
    ap.add_argument("--winner-learning-rate", type=float, default=3e-4)
    ap.add_argument("--winner-epochs", type=int, default=16)
    ap.add_argument("--winner-refresh-every", type=int, default=5)
    ap.add_argument("--winner-refresh-epochs", type=int, default=1)
    ap.add_argument("--winner-amplify-rounds", type=int, default=3)
    ap.add_argument("--winner-recheck-every", type=int, default=10)
    ap.add_argument("--winner-max-checks", type=int, default=3)
    ap.add_argument("--rollback-margin", type=float, default=3500.0)
    ap.add_argument("--migration-grace-iters", type=int, default=30)
    ap.add_argument("--gamma", type=float, default=0.985)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-ratio", type=float, default=0.15)
    ap.add_argument("--entropy-coef", type=float, default=0.030)
    ap.add_argument("--value-coef", type=float, default=0.30)
    ap.add_argument("--target-kl", type=float, default=0.015)
    ap.add_argument("--micro-learning-rate", type=float, default=0.000125)
    ap.add_argument("--micro-gamma", type=float, default=0.97)
    ap.add_argument("--micro-gae-lambda", type=float, default=0.90)
    ap.add_argument("--micro-ppo-epochs", type=int, default=1)
    ap.add_argument("--micro-clip-ratio", type=float, default=0.08)
    ap.add_argument("--micro-entropy-coef", type=float, default=0.010)
    ap.add_argument("--micro-value-coef", type=float, default=0.20)
    ap.add_argument("--micro-target-kl", type=float, default=0.010)
    ap.add_argument("--skill-cutover-start", type=int, default=720)
    ap.add_argument("--skill-cutover-min", type=int, default=720)
    ap.add_argument("--skill-cutover-step-size", type=int, default=24)
    ap.add_argument("--skill-shadow-start-step", type=int, default=0)
    ap.add_argument("--skill-confidence-threshold", type=float, default=0.70)
    ap.add_argument("--skill-min-takeover-ratio", type=float, default=0.08)
    ap.add_argument("--skill-min-executed-samples", type=int, default=32)
    ap.add_argument("--skill-keep-penalty", type=float, default=2.75)
    ap.add_argument("--skill-bc-epochs", type=int, default=4)
    ap.add_argument("--skill-teacher-per-class", type=int, default=2048)
    ap.add_argument("--skill-prior-power", type=float, default=0.85)
    ap.add_argument("--skill-bc-min-nonkeep-acc", type=float, default=0.72)
    ap.add_argument("--skill-bc-min-core-acc", type=float, default=0.50)
    ap.add_argument("--skill-val-min-core-f1", type=float, default=0.70)
    ap.add_argument("--skill-stable-evals-required", type=int, default=3)
    ap.add_argument("--skill-floor-eval-every", type=int, default=2)
    ap.add_argument("--skill-stage-margin-tolerance", type=float, default=1000.0)
    ap.add_argument("--skill-max-plant-deaths-per-game", type=float, default=20.0)
    ap.add_argument("--skill-max-animal-escapes", type=int, default=0)
    ap.add_argument("--init-checkpoint", default="")
    ap.add_argument("--init-elite-memory", default="")
    ap.add_argument(
        "--import-v3-output",
        default="",
        help="V3 output directory containing elite_v3_best.pt and elite_memory_v3.pt",
    )
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--max-iterations", type=int, default=0, help="0 = run until Ctrl+C")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    # Respect the requested curriculum floor. 720 is shadow-only because
    # Kaggriculture frames are 0..719; forcing min>=720 permanently prevents
    # full-skill execution and leaves skill_records/active_ratio at zero.
    args.skill_cutover_min = max(
        0, min(int(args.skill_cutover_start), int(args.skill_cutover_min))
    )

    import kaggle_environments as vendored_kaggle
    engine_path = Path(vendored_kaggle.__file__).resolve()
    expected_vendor = (ROOT / "vendor").resolve()
    if expected_vendor not in engine_path.parents:
        raise RuntimeError(
            f"V4 must use bundled Kaggriculture engine, got {engine_path}"
        )
    print(
        f"ENGINE={vendored_kaggle.__version__} path={engine_path}",
        flush=True,
    )

    if not torch.cuda.is_available() and not args.smoke:
        raise RuntimeError(
            "CUDA GPU is required for continuous training. "
            "Use --smoke only for package verification."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        props = torch.cuda.get_device_properties(0)
        print(
            f"GPU={props.name} VRAM={props.total_memory/(1024**3):.2f}GB "
            f"torch={torch.__version__}",
            flush=True,
        )
    else:
        print("CPU smoke mode", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / "winner_v4_latest.pt"
    best = out_dir / "winner_v4_best.pt"
    winner_anchor = out_dir / "winner_v4_reproduced.pt"
    safe_best = out_dir / "winner_v4_safe_best.pt"
    skill_stage_safe = out_dir / "winner_v45_skill_stage_safe.pt"
    snapshot = out_dir / "_rollout_actor_v4.pt"
    csv_path = out_dir / "winner_v4_metrics.csv"
    elite_path = out_dir / "winner_memory_v4.pt"
    skill_teacher_path = out_dir / "skill_teacher_train_v46.pt"
    skill_teacher_val_path = out_dir / "skill_teacher_val_v46.pt"

    if args.reset:
        for stale in (
            latest,
            best,
            snapshot,
            csv_path,
            elite_path,
            skill_teacher_path,
            skill_teacher_val_path,
            winner_anchor,
            safe_best,
            skill_stage_safe,
            out_dir / "winner_v4_actor_last.pt",
            out_dir / "winner_v4_summary.json",
            out_dir / "baseline_eval.json",
        ):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass

    if args.import_v3_output:
        v3_dir = Path(args.import_v3_output).expanduser()
        if not args.init_checkpoint:
            for candidate in (
                v3_dir / "elite_v3_best.pt",
                v3_dir / "elite_v3_latest.pt",
            ):
                if candidate.exists():
                    args.init_checkpoint = str(candidate)
                    break
        if not args.init_elite_memory:
            candidate = v3_dir / "elite_memory_v3.pt"
            if candidate.exists():
                args.init_elite_memory = str(candidate)
        print(
            f"IMPORT_V3 checkpoint={args.init_checkpoint or 'NONE'} "
            f"elite_memory={args.init_elite_memory or 'NONE'}",
            flush=True,
        )

    payload, parent = load_parent(args.parent, device=device)
    route_count = len(payload["route_ids"])
    market_count = len(payload["market_modes"])
    actor = ContinuousActor(
        int(payload["hidden_dim"]),
        int(payload["clock_dim"]),
        route_count,
        market_count,
    ).to(device)
    init_actor(actor)
    freeze_micro_representation = (
        os.environ.get("FARMOS_FREEZE_MICRO_REP", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if freeze_micro_representation:
        for parameter in actor.micro_trunk.parameters():
            parameter.requires_grad_(False)
        for parameter in actor.micro_plant_context.parameters():
            parameter.requires_grad_(False)
        print(
            "ACT_KEEP freeze_micro_representation=1 "
            "(micro_trunk + micro_plant_context)",
            flush=True,
        )
    micro_prefixes = (
        "micro_trunk.", "micro_plant_context.", "micro_task.", "micro_value."
    )
    harvest_prefix = "harvest_router."
    macro_params = [
        parameter
        for name, parameter in actor.named_parameters()
        if not name.startswith(micro_prefixes)
        and not name.startswith(harvest_prefix)
    ]
    micro_params = [
        parameter
        for name, parameter in actor.named_parameters()
        if name.startswith(micro_prefixes)
    ]
    harvest_params = [
        parameter
        for name, parameter in actor.named_parameters()
        if name.startswith(harvest_prefix)
    ]
    optimizer = torch.optim.AdamW(
        macro_params, lr=args.learning_rate, weight_decay=1e-5
    )
    micro_optimizer = torch.optim.AdamW(
        micro_params, lr=args.micro_learning_rate, weight_decay=1e-5
    )
    harvest_optimizer = torch.optim.AdamW(
        harvest_params, lr=args.harvest_learning_rate, weight_decay=1e-5
    )
    conditional_prefixes = (
        "route_action_embedding.",
        "market_action_embedding.",
        "market_condition.",
        "horizon_condition.",
        "economic_router.",
        "opponent_router.",
    )
    conditional_params = []
    legacy_params = []
    for name, parameter in actor.named_parameters():
        if name.startswith(micro_prefixes) or name.startswith(harvest_prefix):
            continue
        if name.startswith(conditional_prefixes):
            conditional_params.append(parameter)
        else:
            legacy_params.append(parameter)
    winner_optimizer = torch.optim.AdamW(
        [
            {
                "params": legacy_params,
                "lr": args.winner_learning_rate * 0.25,
            },
            {
                "params": conditional_params,
                "lr": args.winner_learning_rate,
            },
        ],
        weight_decay=0.0,
    )
    start_iteration = 0
    best_margin = -float("inf")
    best_win_rate = 0.0
    winner_reproduced = False
    verified_winner_margin = None
    verified_winner_seed = None
    verified_winner_seat = None
    verified_winner_game = None
    winner_last_recheck_iteration = -1
    migrated_new_params = False
    current_skill_cutover = max(
        int(args.skill_cutover_min), int(args.skill_cutover_start)
    )
    skill_stable_evals = 0
    skill_stage_reference_margin = None
    skill_stage_reference_cutover = None

    if latest.exists() and not args.reset:
        state = torch.load(latest, map_location=device, weights_only=False)
        load_info = load_actor_state_compatible(actor, state["actor_state"])
        source_schema = str(state.get("schema", ""))
        non_harvest_missing = [
            key for key in load_info["missing"]
            if not str(key).startswith("harvest_router.")
        ]
        harvest_only_migration = bool(load_info["missing"]) and not non_harvest_missing and not load_info.get("partial")
        migrated_new_params = bool(
            source_schema != "farmos_v51_winner_amplification_state_v4_5_skill_manager"
            or non_harvest_missing
            or load_info.get("partial")
        )
        if harvest_only_migration:
            print(
                f"MIGRATE_V4_6_HARVEST_ONLY new_params={len(load_info['missing'])} "
                "preserve_macro_optimizer=1",
                flush=True,
            )
        if migrated_new_params:
            print(
                f"MIGRATE_TO_V4_5 source={source_schema or 'UNKNOWN'} "
                f"matched={load_info['matched']} new_params={len(load_info['missing'])} "
                f"partial={len(load_info.get('partial', []))}",
                flush=True,
            )
        if migrated_new_params:
            optimizer.state.clear()
            print(
                "RESET_PPO_OPTIMIZER V4.5 migration changed parameter shapes",
                flush=True,
            )
        else:
            try:
                optimizer.load_state_dict(state["optimizer_state"])
            except Exception:
                optimizer.state.clear()
                print(
                    "RESET_PPO_OPTIMIZER state incompatible with current parameter set",
                    flush=True,
                )
        if migrated_new_params:
            micro_optimizer.state.clear()
            print(
                "RESET_MICRO_OPTIMIZER V4.5 migration changed micro parameter shapes",
                flush=True,
            )
        elif isinstance(state.get("micro_optimizer_state"), dict):
            try:
                micro_optimizer.load_state_dict(state["micro_optimizer_state"])
            except Exception:
                micro_optimizer.state.clear()
                print(
                    "RESET_MICRO_OPTIMIZER state incompatible with current micro parameters",
                    flush=True,
                )
        # Optimizer state_dict stores param-group LR. Re-apply requested LRs
        # so legacy checkpoints cannot silently override conservative settings.
        for group in micro_optimizer.param_groups:
            group["lr"] = float(args.micro_learning_rate)
        if isinstance(state.get("harvest_optimizer_state"), dict):
            try:
                harvest_optimizer.load_state_dict(state["harvest_optimizer_state"])
            except Exception:
                harvest_optimizer.state.clear()
                print("RESET_HARVEST_OPTIMIZER incompatible state", flush=True)
        else:
            harvest_optimizer.state.clear()
        for group in harvest_optimizer.param_groups:
            group["lr"] = float(args.harvest_learning_rate)
        start_iteration = int(state.get("iteration", 0))
        best_margin = float(state.get("best_margin", -float("inf")))
        best_win_rate = float(state.get("best_win_rate", 0.0))
        winner_reproduced = bool(state.get("winner_reproduced", False))
        verified_winner_margin = state.get("verified_winner_margin")
        verified_winner_seed = state.get("verified_winner_seed")
        verified_winner_seat = state.get("verified_winner_seat")
        winner_last_recheck_iteration = int(
            state.get("winner_last_recheck_iteration", -1)
        )
        current_skill_cutover = int(
            state.get("skill_cutover_step", current_skill_cutover)
        )
        current_skill_cutover = max(
            int(args.skill_cutover_min),
            min(int(args.skill_cutover_start), current_skill_cutover),
        )
        skill_stable_evals = int(state.get("skill_stable_evals", 0))
        skill_stage_reference_margin = state.get("skill_stage_reference_margin")
        skill_stage_reference_cutover = state.get("skill_stage_reference_cutover")
        if (
            skill_stage_reference_margin is None
            and skill_stage_safe.exists()
        ):
            try:
                _stage_state = torch.load(
                    skill_stage_safe, map_location="cpu", weights_only=False
                )
                skill_stage_reference_margin = _stage_state.get(
                    "skill_stage_reference_margin"
                )
                skill_stage_reference_cutover = _stage_state.get(
                    "skill_stage_reference_cutover"
                )
            except Exception:
                pass
        if migrated_new_params:
            winner_optimizer.state.clear()
            print(
                "RESET_WINNER_OPTIMIZER V4.5 migration changed parameter shapes",
                flush=True,
            )
        elif isinstance(state.get("winner_optimizer_state"), dict):
            try:
                winner_optimizer.load_state_dict(state["winner_optimizer_state"])
            except Exception:
                winner_optimizer.state.clear()
                print(
                    "RESET_WINNER_OPTIMIZER state incompatible with current parameter groups",
                    flush=True,
                )
        print(
            f"RESUME iteration={start_iteration} "
            f"best_margin={best_margin:+.1f} best_win_rate={100*best_win_rate:.2f}% "
            f"winner_reproduced={winner_reproduced}",
            flush=True,
        )
    elif args.init_checkpoint:
        if load_v1_warmstart(actor, args.init_checkpoint):
            print(f"WARMSTART loaded from {args.init_checkpoint}", flush=True)

    if (not elite_path.exists()) and args.init_elite_memory:
        source_memory = Path(args.init_elite_memory).expanduser()
        if not source_memory.exists():
            raise FileNotFoundError(f"init elite memory not found: {source_memory}")
        memory_payload = torch.load(
            source_memory, map_location="cpu", weights_only=False
        )
        imported_games = list(memory_payload.get("games", []))
        torch.save(
            {
                "schema": "farmos_v51_winner_memory_v4_3_context",
                "games": imported_games,
                "imported_from": str(source_memory),
            },
            elite_path,
        )
        print(
            f"IMPORTED_ELITE_MEMORY games={len(imported_games)} "
            f"from={source_memory}",
            flush=True,
        )

    if args.smoke:
        args.workers = max(1, min(args.workers, 2))
        args.train_seeds_per_iter = 1
        args.eval_every = 1
        args.eval_seeds = f"{args.seed_base+999}:1"
        args.ppo_epochs = 1
        args.max_iterations = 1

    config = vars(args).copy()
    config["gpu"] = (
        torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU_SMOKE"
    )
    config["run_mode"] = "winner_amplification_exact_v51_v4_5_learned_skill_curriculum_parallel32"
    config["economic_context_dim"] = ECONOMIC_DIM
    config["opponent_context_dim"] = OPPONENT_DIM
    config["dense_farm_reward"] = True
    config["profit_chain_reward"] = True
    config["plant_death_penalty"] = -1.5
    config["animal_escape_penalty"] = -4.0
    config["route_handoff_mask"] = "opening_only_then_one_way_skill_cutover"
    config["skill_manager"] = "learned_masked_full_worker_ppo"
    config["skill_cutover_initial"] = int(current_skill_cutover)
    config["state_safe_farm_supervisor"] = False
    config["safe_idle_labor"] = False
    config["worker_micro_policy"] = "retired_hands_pre_cutover_full_workers_post_cutover"
    config["micro_tasks"] = list(MICRO_TASKS)
    config["context_aware_replay"] = True
    config["structural_gate"] = "exact_gain_table_v4_2"
    config["structural_source_routes"] = sorted(COW_BRANCH_SOURCE_ROUTES)
    config["structural_gain_table_schema"] = STRUCTURAL_GAIN_TABLE.get("schema")
    config["idle_weed_labor"] = True
    config["stop_condition"] = "Ctrl+C"
    config["validation_engine"] = "vendored kaggriculture 1.32.7 core"
    migration_guard_until = (
        start_iteration + max(0, int(args.migration_grace_iters))
        if migrated_new_params
        else start_iteration
    )
    if migrated_new_params and args.migration_grace_iters > 0:
        print(
            f"[MIGRATION-GRACE] eval rollback disabled through "
            f"iter={migration_guard_until}; winner recheck remains active",
            flush=True,
        )

    eval_seeds = parse_seed_range(args.eval_seeds)
    actor.eval()

    rollout_pool = None
    if args.workers > 1:
        ctx = mp.get_context("spawn")
        rollout_pool = ctx.Pool(
            processes=args.workers,
            maxtasksperchild=(
                args.worker_recycle_tasks
                if args.worker_recycle_tasks > 0
                else None
            ),
        )

    print(
        "WINNER V4.5 LEARNED-SKILL CURRICULUM TRAINING STARTED - press Ctrl+C to stop safely.",
        flush=True,
    )
    print(
        f"PARALLEL rollout_workers={args.workers} "
        f"train_games_per_iter={args.train_seeds_per_iter} "
        f"worker_recycle_tasks={args.worker_recycle_tasks} "
        f"logical_cpus={os.cpu_count()}",
        flush=True,
    )
    print(
        "Money/margin below come from exact 720-step games vs v51.",
        flush=True,
    )
    print(
        f"SKILL curriculum cutover={current_skill_cutover} "
        f"min={args.skill_cutover_min} step_size={args.skill_cutover_step_size} "
        f"stable_evals={skill_stable_evals}/{args.skill_stable_evals_required} "
        f"takeover_min={args.skill_min_takeover_ratio:.2f} "
        f"exec_min={args.skill_min_executed_samples} "
        f"keep_penalty={args.skill_keep_penalty:.2f}",
        flush=True,
    )
    print(
        f"MICRO cfg lr={args.micro_learning_rate:.6g} "
        f"epochs={args.micro_ppo_epochs} clip={args.micro_clip_ratio:.3f} "
        f"entropy={args.micro_entropy_coef:.3f} target_kl={args.micro_target_kl:.3f} "
        f"floor_eval_every={args.skill_floor_eval_every}",
        flush=True,
    )

    if start_iteration == 0 and not args.smoke:
        save_snapshot(
            actor,
            snapshot,
            {"iteration": 0, "temperature": args.min_temperature},
        )
        baseline_rows = exact_games(
            args.parent,
            snapshot,
            args.opponent,
            eval_seeds,
            args.workers,
            False,
            args.min_temperature,
            args.residual_scale,
            args.base_keep_bias,
            args.decision_every,
            0,
            pool=rollout_pool,
            both_seats=True,
            skill_cutover_step=current_skill_cutover,
            skill_keep_penalty=args.skill_keep_penalty,
            skill_shadow_start_step=args.skill_shadow_start_step,
            skill_confidence_threshold=args.skill_confidence_threshold,
        )
        baseline_m = metrics(baseline_rows)
        best_win_rate = baseline_m["win_rate"]
        best_margin = baseline_m["mean_margin"]
        print_metrics("BASE ", 0, baseline_m, best=best_margin)
        save_state(
            best,
            actor,
            optimizer,
            0,
            best_margin,
            best_win_rate,
            config,
        )
        (out_dir / "baseline_eval.json").write_text(
            json.dumps(
                {"summary": baseline_m, "games": baseline_rows},
                indent=2,
                default=lambda x: "<binary>" if isinstance(x, bytes) else x,
            ),
            encoding="utf-8",
        )

    def winner_state_extra():
        return {
            "winner_reproduced": bool(winner_reproduced),
            "verified_winner_margin": verified_winner_margin,
            "verified_winner_seed": verified_winner_seed,
            "verified_winner_seat": verified_winner_seat,
            "winner_last_recheck_iteration": int(winner_last_recheck_iteration),
            "skill_cutover_step": int(current_skill_cutover),
            "skill_stable_evals": int(skill_stable_evals),
            "skill_stage_reference_margin": skill_stage_reference_margin,
            "skill_stage_reference_cutover": skill_stage_reference_cutover,
            "micro_optimizer_state": micro_optimizer.state_dict(),
            "harvest_optimizer_state": harvest_optimizer.state_dict(),
            "winner_optimizer_state": winner_optimizer.state_dict(),
        }

    startup_elite_games = []
    if elite_path.exists():
        try:
            startup_elite_games = list(
                torch.load(
                    elite_path, map_location="cpu", weights_only=False
                ).get("games", [])
            )
        except Exception:
            startup_elite_games = []

    startup_ws = winner_summary(startup_elite_games)
    if startup_ws["winner_count"] > 0:
        verified_winner_game, scripted_replay, scripted_failures = pick_verified_winner(
            startup_elite_games,
            args.parent,
            args.opponent,
            max_checks=args.winner_max_checks,
        )
        if verified_winner_game is not None:
            verified_winner_margin = int(verified_winner_game["margin"])
            verified_winner_seed = int(verified_winner_game["seed"])
            verified_winner_seat = int(verified_winner_game["seat"])
            print(
                f"[WINNER-VERIFY] PASS seed={verified_winner_seed} "
                f"seat={verified_winner_seat} "
                f"historical_margin={verified_winner_margin:+d} "
                f"runtime_margin={int(scripted_replay['margin']):+d} "
                f"historical_match={bool(scripted_replay['exact_margin_match'])} "
                f"decisions={scripted_replay['applied_decisions']}",
                flush=True,
            )
            startup_amp = amplify_single_winner(
                actor,
                parent,
                winner_optimizer,
                startup_elite_games,
                verified_winner_game,
                snapshot,
                args.parent,
                args.opponent,
                hidden_dim=int(payload["hidden_dim"]),
                clock_dim=int(payload["clock_dim"]),
                route_count=route_count,
                market_count=market_count,
                device=device,
                residual_scale=args.residual_scale,
                base_keep_bias=args.base_keep_bias,
                decision_every=args.decision_every,
                temperature=args.min_temperature,
                epochs=args.winner_epochs,
                rounds=args.winner_amplify_rounds,
                minibatch=args.minibatch,
                iteration=start_iteration,
            )
            winner_reproduced = bool(startup_amp["reproduced"])
            winner_last_recheck_iteration = start_iteration
            print(
                f"[WINNER-GATE] startup reproduced={winner_reproduced} "
                f"actor_margin={startup_amp.get('actor_margin')} "
                f"reason={startup_amp['reason']}",
                flush=True,
            )
            if winner_reproduced:
                save_state(
                    winner_anchor,
                    actor,
                    optimizer,
                    start_iteration,
                    best_margin,
                    best_win_rate,
                    config,
                    extra=winner_state_extra(),
                )
        else:
            winner_reproduced = False
            print(
                f"[WINNER-VERIFY] FAIL checked={len(scripted_failures)} "
                "stored winners; exploration will continue until a verifiable winner exists.",
                flush=True,
            )

    iteration = start_iteration
    completed_this_run = 0
    try:
        while True:
            if args.max_iterations and completed_this_run >= args.max_iterations:
                break
            iteration += 1
            completed_this_run += 1
            harvest_only_warmup = (
                int(args.harvest_only_warmup_iters) > 0
                and completed_this_run <= int(args.harvest_only_warmup_iters)
            )
            if harvest_only_warmup:
                print(
                    f"[V4.6-WARMUP] iter={iteration:05d} harvest_only=1 "
                    f"step={completed_this_run}/{int(args.harvest_only_warmup_iters)}",
                    flush=True,
                )

            # Slowly reduce exploration, but keep a floor so training never becomes
            # fully deterministic while the process is running.
            decay = math.exp(-iteration / 250.0)
            temperature = (
                args.min_temperature
                + (args.temperature - args.min_temperature) * decay
            )

            if (
                verified_winner_game is not None
                and not winner_reproduced
                and not harvest_only_warmup
            ):
                print(
                    f"[MODE] iter={iteration:05d} AMPLIFY_ONLY "
                    f"seed={verified_winner_seed} seat={verified_winner_seat} "
                    f"teacher_margin={int(verified_winner_margin):+d}",
                    flush=True,
                )
                current_games = torch.load(
                    elite_path, map_location="cpu", weights_only=False
                ).get("games", [])
                amp = amplify_single_winner(
                    actor,
                    parent,
                    winner_optimizer,
                    current_games,
                    verified_winner_game,
                    snapshot,
                    args.parent,
                    args.opponent,
                    hidden_dim=int(payload["hidden_dim"]),
                    clock_dim=int(payload["clock_dim"]),
                    route_count=route_count,
                    market_count=market_count,
                    device=device,
                    residual_scale=args.residual_scale,
                    base_keep_bias=args.base_keep_bias,
                    decision_every=args.decision_every,
                    temperature=args.min_temperature,
                    epochs=args.winner_epochs,
                    rounds=args.winner_amplify_rounds,
                    minibatch=args.minibatch,
                    iteration=iteration,
                )
                winner_reproduced = bool(amp["reproduced"])
                winner_last_recheck_iteration = iteration
                if winner_reproduced:
                    save_state(
                        winner_anchor,
                        actor,
                        optimizer,
                        iteration,
                        best_margin,
                        best_win_rate,
                        config,
                        extra=winner_state_extra(),
                    )
                    print(
                        f"[WINNER-GATE] PASS actor_margin={amp['actor_margin']:+d}; "
                        "fresh exploration unlocked.",
                        flush=True,
                    )
                else:
                    save_state(
                        latest,
                        actor,
                        optimizer,
                        iteration,
                        best_margin,
                        best_win_rate,
                        config,
                        extra=winner_state_extra(),
                    )
                    print(
                        f"[WINNER-GATE] FAIL actor_margin={amp.get('actor_margin')}; "
                        "fresh exploration remains paused.",
                        flush=True,
                    )
                    continue

            if (
                verified_winner_game is not None
                and winner_reproduced
                and iteration - winner_last_recheck_iteration
                >= args.winner_recheck_every
            ):
                save_snapshot(
                    actor,
                    snapshot,
                    {"iteration": iteration, "purpose": "winner_recheck"},
                )
                recheck = run_actor_reproduction(
                    verified_winner_game,
                    args.parent,
                    snapshot,
                    args.opponent,
                    temperature=args.min_temperature,
                    residual_scale=args.residual_scale,
                    base_keep_bias=args.base_keep_bias,
                    decision_every=args.decision_every,
                    iteration=iteration,
                )
                winner_last_recheck_iteration = iteration
                if recheck["margin"] > 0:
                    save_state(
                        winner_anchor,
                        actor,
                        optimizer,
                        iteration,
                        best_margin,
                        best_win_rate,
                        config,
                        extra=winner_state_extra(),
                    )
                    print(
                        f"[WINNER-RECHECK] PASS seed={verified_winner_seed} "
                        f"seat={verified_winner_seat} margin={recheck['margin']:+d} "
                        f"anchor_refreshed_iter={iteration}",
                        flush=True,
                    )
                else:
                    winner_reproduced = False
                    print(
                        f"[WINNER-RECHECK] FAIL margin={recheck['margin']:+d}; "
                        "restoring known winner behavior.",
                        flush=True,
                    )
                    if winner_anchor.exists():
                        anchor = torch.load(
                            winner_anchor, map_location=device, weights_only=False
                        )
                        # Reinitialize migration-only parameters before loading.
                        # If the anchor predates V4.3, the economic branch must
                        # roll back to its neutral zero residual as well.
                        init_actor(actor)
                        load_actor_state_compatible(
                            actor, anchor["actor_state"]
                        )
                        try:
                            optimizer.load_state_dict(anchor["optimizer_state"])
                        except Exception:
                            optimizer.state.clear()
                        if isinstance(anchor.get("micro_optimizer_state"), dict):
                            try:
                                micro_optimizer.load_state_dict(
                                    anchor["micro_optimizer_state"]
                                )
                            except Exception:
                                micro_optimizer.state.clear()
                        else:
                            micro_optimizer.state.clear()
                        for group in micro_optimizer.param_groups:
                            group["lr"] = float(args.micro_learning_rate)
                        if isinstance(anchor.get("harvest_optimizer_state"), dict):
                            try:
                                harvest_optimizer.load_state_dict(anchor["harvest_optimizer_state"])
                            except Exception:
                                harvest_optimizer.state.clear()
                        else:
                            harvest_optimizer.state.clear()
                        for group in harvest_optimizer.param_groups:
                            group["lr"] = float(args.harvest_learning_rate)
                        if isinstance(anchor.get("winner_optimizer_state"), dict):
                            try:
                                winner_optimizer.load_state_dict(
                                    anchor["winner_optimizer_state"]
                                )
                            except Exception:
                                winner_optimizer.state.clear()
                        winner_reproduced = True
                        print(
                            "[WINNER-ROLLBACK] restored winner_v4_reproduced.pt",
                            flush=True,
                        )
                    else:
                        save_state(
                            latest,
                            actor,
                            optimizer,
                            iteration,
                            best_margin,
                            best_win_rate,
                            config,
                            extra=winner_state_extra(),
                        )
                        continue

            save_snapshot(
                actor,
                snapshot,
                {
                    "iteration": iteration,
                    "temperature": temperature,
                },
            )
            train_seed_start = (
                args.seed_base
                + (iteration - 1) * args.train_seeds_per_iter
            )
            train_seeds = list(
                range(
                    train_seed_start,
                    train_seed_start + args.train_seeds_per_iter,
                )
            )
            t0 = time.time()
            rows = exact_games(
                args.parent,
                snapshot,
                args.opponent,
                train_seeds,
                args.workers,
                True,
                temperature,
                args.residual_scale,
                args.base_keep_bias,
                args.decision_every,
                iteration,
                pool=rollout_pool,
                both_seats=False,
                skill_cutover_step=current_skill_cutover,
                skill_keep_penalty=args.skill_keep_penalty,
                skill_shadow_start_step=args.skill_shadow_start_step,
                skill_confidence_threshold=args.skill_confidence_threshold,
            )
            train_m = metrics(rows)
            print_metrics(
                "TRAIN",
                iteration,
                train_m,
                best=best_margin if math.isfinite(best_margin) else None,
                elapsed=time.time() - t0,
            )
            econ_s = economic_rollout_summary(rows)
            if econ_s is not None:
                print(
                    f"[ECON ] iter={iteration:05d} context_games={econ_s['context_games']} "
                    f"shops(yarn/pizza)={econ_s['yarn_shops']:.2f}/{econ_s['pizza_shops']:.2f} "
                    f"roi(goose/cow/sheep)={econ_s['goose_roi']:+.3f}/"
                    f"{econ_s['cow_roi']:+.3f}/{econ_s['sheep_roi']:+.3f} "
                    f"crop_best={econ_s['crop_best']}:{econ_s['crop_best_score']:+.3f}",
                    flush=True,
                )
            print(
                f"[ROUTE] iter={iteration:05d} "
                f"handoff_candidates={train_m['handoff_candidates_mean']:.2f} "
                f"locked_decisions={train_m['handoff_locked']} replans={train_m['shop_replans']}",
                flush=True,
            )
            print(
                f"[CARE ] iter={iteration:05d} dense={train_m['dense_reward']:+.3f}/game "
                f"death(plant/animal)={train_m['plant_deaths']}/{train_m['animal_escapes']} "
                f"done(water/feed/care/fert)={train_m['watered']}/{train_m['fed']}/"
                f"{train_m['cared']}/{train_m['fertilized']} "
                f"chain(collect/harvest/place/sell)={train_m['fertilizer_collected']}/"
                f"{train_m['harvested_units']}/{train_m['placed_units']}/{train_m['sold_units']} "
                f"create(plant/structure/animal)={train_m['plants_created']}/"
                f"{train_m['structures_built']}/{train_m['animals_placed']} "
                f"micro_records={train_m['micro_records']} invalid_ops={train_m['invalid_ops']}",
                flush=True,
            )
            water_exec_rate = (
                train_m['water_success_actions'] / max(1, train_m['water_attempts'])
            )
            fert_exec_rate = (
                train_m['fert_collect_success_actions'] / max(1, train_m['fert_collect_attempts'])
            )
            print(
                f"[CARE-EXEC] iter={iteration:05d} "
                f"water(critical/attempt/success/rate)="
                f"{train_m['critical_water_opportunities']}/{train_m['water_attempts']}/"
                f"{train_m['water_success_actions']}/{100*water_exec_rate:.1f}% "
                f"collect_fert(available/attempt/success/rate)="
                f"{train_m['fert_collect_opportunities']}/{train_m['fert_collect_attempts']}/"
                f"{train_m['fert_collect_success_actions']}/{100*fert_exec_rate:.1f}%",
                flush=True,
            )

            batch = records_to_training(
                rows,
                int(payload["hidden_dim"]),
                int(payload["clock_dim"]),
                device,
                args.gamma,
                args.gae_lambda,
            )
            ppo = ppo_update(
                actor,
                parent,
                batch,
                optimizer,
                harvest_optimizer,
                route_count,
                market_count,
                residual_scale=args.residual_scale,
                base_keep_bias=args.base_keep_bias,
                temperature=temperature,
                epochs=args.ppo_epochs,
                minibatch=args.minibatch,
                clip_ratio=args.clip_ratio,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
                target_kl=args.target_kl,
                update_macro=not harvest_only_warmup,
            )
            print(
                f"[PPO]   iter={iteration:05d} "
                f"policy={ppo['policy_loss']:+.5f} "
                f"value={ppo['value_loss']:.5f} "
                f"entropy={ppo['entropy']:.4f} "
                f"kl={ppo['approx_kl']:+.5f} "
                f"clip={ppo['clip_fraction']:.3f} "
                f"temp={temperature:.3f}",
                flush=True,
            )
            with torch.no_grad():
                _harvest_w = actor.harvest_router[-1].weight.detach()
                _harvest_b = actor.harvest_router[-1].bias.detach()
                print(
                    f"[HARVEST] iter={iteration:05d} dim={HARVEST_ECON_DIM} "
                    f"lr={args.harvest_learning_rate:.2e} "
                    f"w_norm={float(_harvest_w.norm()):.6f} "
                    f"w_max={float(_harvest_w.abs().max()):.6f} "
                    f"b_max={float(_harvest_b.abs().max()):.6f}",
                    flush=True,
                )

            micro_batch = micro_records_to_training(
                rows,
                int(payload["hidden_dim"]),
                int(payload["clock_dim"]),
                device,
                gamma=args.micro_gamma,
                lam=args.micro_gae_lambda,
            )
            micro = micro_ppo_update(
                actor,
                micro_batch,
                micro_optimizer,
                temperature=temperature,
                epochs=args.micro_ppo_epochs,
                minibatch=args.minibatch,
                clip_ratio=args.micro_clip_ratio,
                entropy_coef=args.micro_entropy_coef,
                value_coef=args.micro_value_coef,
                target_kl=args.micro_target_kl,
                skill_keep_penalty=args.skill_keep_penalty,
                update=not harvest_only_warmup,
            )
            task_mix = "/".join(
                f"{MICRO_TASKS[i]}:{int(count)}"
                for i, count in enumerate(micro["task_counts"])
                if int(count) > 0
            ) or "none"
            print(
                f"[MICRO] iter={iteration:05d} rows={micro['rows']:5d} "
                f"policy={micro['policy_loss']:+.5f} value={micro['value_loss']:.5f} "
                f"entropy={micro['entropy']:.4f} kl={micro['approx_kl']:+.5f} "
                f"clip={micro['clip_fraction']:.3f} tasks={task_mix}",
                flush=True,
            )

            skill_train_rows, skill_val_rows = split_skill_teacher_rows(rows)
            skill_natural_counts = skill_teacher_class_counts(skill_train_rows)
            skill_teacher_records, skill_replay_counts = update_skill_teacher_memory(
                skill_teacher_path,
                skill_train_rows,
                per_class_limit=args.skill_teacher_per_class,
                iteration=iteration,
            )
            skill_val_records, skill_val_replay_counts = update_skill_teacher_memory(
                skill_teacher_val_path,
                skill_val_rows,
                per_class_limit=max(128, int(args.skill_teacher_per_class) // 4),
                iteration=iteration,
            )
            skill_teacher_batch = skill_teacher_records_to_batch(
                [{"skill_teacher_records": skill_teacher_records}],
                int(payload["hidden_dim"]),
                int(payload["clock_dim"]),
                device,
                natural_counts=skill_natural_counts,
                prior_power=float(args.skill_prior_power),
            )
            skill_bc = skill_bc_update(
                actor,
                skill_teacher_batch,
                micro_optimizer,
                temperature=temperature,
                epochs=args.skill_bc_epochs,
                minibatch=args.minibatch,
                keep_penalty=args.skill_keep_penalty,
                update=not harvest_only_warmup,
            )
            skill_val_batch = skill_teacher_records_to_batch(
                [{"skill_teacher_records": skill_val_records}],
                int(payload["hidden_dim"]),
                int(payload["clock_dim"]),
                device,
            )
            skill_val = skill_bc_update(
                actor,
                skill_val_batch,
                micro_optimizer,
                temperature=temperature,
                epochs=0,
                minibatch=args.minibatch,
                keep_penalty=args.skill_keep_penalty,
                update=False,
            )
            skill_teacher_mix = "/".join(
                f"{MICRO_TASKS[i]}:{int(count)}"
                for i, count in enumerate(skill_bc.get("target_counts", []))
                if int(count) > 0
            ) or "none"
            skill_acc_mix = "/".join(
                f"{MICRO_TASKS[i]}:{100*float(acc):.0f}%"
                for i, acc in enumerate(skill_bc.get("per_task_acc", []))
                if math.isfinite(float(acc))
                and int(skill_bc.get("target_counts", [0]*len(MICRO_TASKS))[i]) >= 32
            ) or "none"
            prior_mix = "/".join(
                f"{MICRO_TASKS[i]}:{int(count)}"
                for i, count in enumerate(skill_natural_counts)
                if int(count) > 0
            ) or "none"
            print(
                f"[SKILL-PRIOR] iter={iteration:05d} natural={prior_mix}",
                flush=True,
            )
            print(
                f"[SKILL-BC] iter={iteration:05d} rows={skill_bc['rows']:5d} "
                f"nonkeep={skill_bc['nonkeep_rows']:5d} loss={skill_bc['loss']:.4f} "
                f"acc={100*skill_bc['acc']:.1f}% "
                f"nonkeep_acc={100*skill_bc['nonkeep_acc']:.1f}% "
                f"core_min={100*skill_bc.get('core_min_acc',0.0):.1f}% "
                f"teacher={skill_teacher_mix} skill_acc={skill_acc_mix}",
                flush=True,
            )
            skill_val_mix = "/".join(
                f"{MICRO_TASKS[i]}:{100*float(acc):.0f}%"
                for i, acc in enumerate(skill_val.get("per_task_acc", []))
                if math.isfinite(float(acc))
                and int(skill_val.get("target_counts", [0]*len(MICRO_TASKS))[i]) >= 16
            ) or "none"
            skill_val_counts = list(skill_val.get("target_counts", []))
            skill_val_core_ready = bool(
                len(skill_val_counts) == len(MICRO_TASKS)
                and all(int(skill_val_counts[i]) >= 64 for i in range(1, len(MICRO_TASKS)))
            )
            print(
                f"[SKILL-VAL] iter={iteration:05d} games={len(skill_val_rows):2d} "
                f"rows={skill_val['rows']:5d} nonkeep={skill_val['nonkeep_rows']:5d} "
                f"acc={100*skill_val['acc']:.1f}% "
                f"nonkeep_acc={100*skill_val['nonkeep_acc']:.1f}% "
                f"core_min={100*skill_val.get('core_min_acc',0.0):.1f}% "
                f"core_f1={100*skill_val.get('core_min_f1',0.0):.1f}% "
                f"ready={int(skill_val_core_ready)} skill_acc={skill_val_mix}",
                flush=True,
            )
            skill_val_f1_mix = "/".join(
                f"{MICRO_TASKS[i]}:{100*float(f1):.0f}%"
                for i, f1 in enumerate(skill_val.get("per_task_f1", []))
                if math.isfinite(float(f1))
                and int(skill_val.get("target_counts", [0]*len(MICRO_TASKS))[i]) >= 16
            ) or "none"
            print(
                f"[SKILL-F1 ] iter={iteration:05d} heldout_f1={skill_val_f1_mix}",
                flush=True,
            )
            conf_rows = []
            for true_idx, row in enumerate(skill_val.get("confusion", [])):
                for pred_idx, count in enumerate(row):
                    if true_idx != pred_idx and int(count) > 0:
                        conf_rows.append((int(count), true_idx, pred_idx))
            conf_rows.sort(reverse=True)
            conf_text = "/".join(
                f"{MICRO_TASKS[t]}->{MICRO_TASKS[p]}:{c}"
                for c, t, p in conf_rows[:8]
            ) or "none"
            print(
                f"[SKILL-CONF] iter={iteration:05d} top={conf_text}",
                flush=True,
            )
            skill_takeover_ratio = (
                train_m["skill_executed_sampled"]
                / max(1, train_m["skill_nonkeep_proposals"])
            )
            skill_keep_ratio = (
                train_m["skill_keep_samples"] / max(1, train_m["skill_samples"])
            )
            print(
                f"[SKILL] iter={iteration:05d} cutover={current_skill_cutover} "
                f"records={train_m['skill_records']} samples={train_m['skill_samples']} "
                f"proposals={train_m['skill_nonkeep_proposals']} "
                f"executed={train_m['skill_executed_sampled']} "
                f"takeover={skill_takeover_ratio:.3f} keep={skill_keep_ratio:.3f} "
                f"stable_evals={skill_stable_evals}/{args.skill_stable_evals_required}",
                flush=True,
            )

            elite_games = update_elite_memory(
                elite_path,
                rows,
                limit=args.elite_limit,
                iteration=iteration,
            )
            elite_s = elite_summary(elite_games)
            winner_s = winner_summary(elite_games)
            elite_batch = elite_records_to_batch(
                elite_games,
                int(payload["hidden_dim"]),
                int(payload["clock_dim"]),
                device,
                args.elite_train_games,
            )
            # Once winners exist, the winner/near/good buffer is the
            # authoritative imitation source. Running generic elite BC in
            # parallel was pulling the policy back toward losing trajectories.
            elite_epochs_now = (
                0 if harvest_only_warmup
                else (0 if winner_s["winner_count"] > 0 else 1)
            )
            if (
                not harvest_only_warmup
                and winner_s["winner_count"] == 0
                and elite_s["elite_best_margin"] >= -8000
            ):
                elite_epochs_now = args.elite_epochs
            elite_fit = elite_bc_update(
                actor,
                parent,
                elite_batch,
                optimizer,
                harvest_optimizer,
                route_count,
                market_count,
                residual_scale=args.residual_scale,
                base_keep_bias=args.base_keep_bias,
                epochs=elite_epochs_now,
                minibatch=args.minibatch,
            )

            winner_batch = winner_records_to_batch(
                elite_games,
                int(payload["hidden_dim"]),
                int(payload["clock_dim"]),
                device,
                max_winners=16,
                max_near=32,
                max_good=48,
                include_other=False,
                winner_only=False,
                context_rows=rows,
            )
            if harvest_only_warmup:
                winner_refresh_epochs = 0
            elif winner_s["winner_count"] > 0:
                refresh_every = max(1, int(args.winner_refresh_every))
                winner_refresh_epochs = (
                    int(args.winner_refresh_epochs)
                    if iteration % refresh_every == 0
                    else 0
                )
            else:
                winner_refresh_epochs = 1
            winner_fit = winner_bc_update(
                actor,
                parent,
                winner_batch,
                winner_optimizer,
                harvest_optimizer,
                route_count,
                market_count,
                residual_scale=args.residual_scale,
                base_keep_bias=args.base_keep_bias,
                epochs=winner_refresh_epochs,
                minibatch=args.minibatch,
            )
            print(
                f"[ELITE] iter={iteration:05d} "
                f"games={elite_s['elite_games']:3d} "
                f"win/near/good={winner_s['winner_count']}/"
                f"{winner_s['near_win_count']}/{winner_s['good_count']} "
                f"ctx={winner_s['context_game_count']} "
                f"best_margin={elite_s['elite_best_margin']:+d} "
                f"median={elite_s['elite_median_margin']:+.1f} "
                f"elite_loss={elite_fit['elite_loss']:.4f} "
                f"winner_loss={winner_fit['winner_loss']:.4f}",
                flush=True,
            )
            print(
                f"[WINNER-BC] iter={iteration:05d} rows={winner_fit['rows']:5d} "
                f"winner_rows={winner_fit['winner_rows']:4d} "
                f"market_rows={winner_fit['market_rows']:4d} "
                f"horizon_rows={winner_fit['horizon_rows']:4d} "
                f"acc(route/market/horizon)="
                f"{100*winner_fit['route_acc']:.1f}/"
                f"{100*winner_fit['market_acc']:.1f}/"
                f"{100*winner_fit['horizon_acc']:.1f}% "
                f"p=({winner_fit['winner_route_prob']:.3f},"
                f"{winner_fit['winner_market_prob']:.3f},"
                f"{winner_fit['winner_horizon_prob']:.3f})",
                flush=True,
            )

            top_winners = partition_elite_games(elite_games)["winner"]
            if top_winners and not harvest_only_warmup:
                top_margin = int(top_winners[0]["margin"])
                is_new_verified_target = (
                    verified_winner_margin is None
                    or top_margin > int(verified_winner_margin)
                )
                if is_new_verified_target:
                    candidate, scripted, failures = pick_verified_winner(
                        elite_games,
                        args.parent,
                        args.opponent,
                        max_checks=args.winner_max_checks,
                    )
                    if candidate is not None:
                        verified_winner_game = candidate
                        verified_winner_margin = int(candidate["margin"])
                        verified_winner_seed = int(candidate["seed"])
                        verified_winner_seat = int(candidate["seat"])
                        winner_reproduced = False
                        print(
                            f"[NEW-WINNER] verified seed={verified_winner_seed} "
                            f"seat={verified_winner_seat} "
                            f"historical_margin={verified_winner_margin:+d} "
                            f"runtime_margin={int(scripted['margin']):+d}",
                            flush=True,
                        )
                        amp = amplify_single_winner(
                            actor,
                            parent,
                            winner_optimizer,
                            harvest_optimizer,
                            elite_games,
                            verified_winner_game,
                            snapshot,
                            args.parent,
                            args.opponent,
                            hidden_dim=int(payload["hidden_dim"]),
                            clock_dim=int(payload["clock_dim"]),
                            route_count=route_count,
                            market_count=market_count,
                            device=device,
                            residual_scale=args.residual_scale,
                            base_keep_bias=args.base_keep_bias,
                            decision_every=args.decision_every,
                            temperature=args.min_temperature,
                            epochs=args.winner_epochs,
                            rounds=args.winner_amplify_rounds,
                            minibatch=args.minibatch,
                            iteration=iteration,
                        )
                        winner_reproduced = bool(amp["reproduced"])
                        winner_last_recheck_iteration = iteration
                        if winner_reproduced:
                            save_state(
                                winner_anchor,
                                actor,
                                optimizer,
                                iteration,
                                best_margin,
                                best_win_rate,
                                config,
                                extra=winner_state_extra(),
                            )
                            print(
                                f"[WINNER-GATE] NEW WINNER REPRODUCED "
                                f"actor_margin={amp['actor_margin']:+d}",
                                flush=True,
                            )
                        else:
                            print(
                                f"[WINNER-GATE] NEW WINNER NOT YET REPRODUCED "
                                f"actor_margin={amp.get('actor_margin')}; "
                                "next iteration will pause exploration.",
                                flush=True,
                            )
                    else:
                        print(
                            f"[NEW-WINNER] verification failed for "
                            f"{len(failures)} stored candidates.",
                            flush=True,
                        )

            evaluated = False
            eval_m = None
            # Once learned skills are live, probe safety frequently at every
            # active cutover stage, not only at the curriculum floor. This
            # catches skill drift/animal escapes before ten more updates accrue.
            skill_floor_probe = (
                current_skill_cutover < int(args.skill_cutover_start)
                and iteration % max(1, int(args.skill_floor_eval_every)) == 0
            )
            if (
                (args.smoke and iteration == 1)
                or iteration % args.eval_every == 0
                or skill_floor_probe
            ):
                save_snapshot(
                    actor,
                    snapshot,
                    {
                        "iteration": iteration,
                        "temperature": args.min_temperature,
                    },
                )
                eval_rows = exact_games(
                    args.parent,
                    snapshot,
                    args.opponent,
                    eval_seeds,
                    args.workers,
                    False,
                    args.min_temperature,
                    args.residual_scale,
                    args.base_keep_bias,
                    args.decision_every,
                    iteration,
                    pool=rollout_pool,
                    both_seats=True,
                    skill_cutover_step=current_skill_cutover,
                    skill_keep_penalty=args.skill_keep_penalty,
                    skill_shadow_start_step=args.skill_shadow_start_step,
                    skill_confidence_threshold=args.skill_confidence_threshold,
                )
                eval_m = metrics(eval_rows)
                evaluated = True
                skill_active_ratio = (
                    eval_m["skill_executed_sampled"]
                    / max(1, eval_m["skill_nonkeep_proposals"])
                )
                shadow_ready = (
                    bool(skill_val_core_ready)
                    and skill_val["nonkeep_rows"] >= 128
                    and skill_val["nonkeep_acc"] >= args.skill_bc_min_nonkeep_acc
                    and skill_val.get("core_min_acc", 0.0) >= args.skill_bc_min_core_acc
                    and skill_val.get("core_min_f1", 0.0) >= args.skill_val_min_core_f1
                )
                # Skill-stage safety is independent from whether the current
                # actor can reproduce a stored scripted winner. The immutable
                # winner anchor is the fallback guard; current skill control is
                # judged by its own margin/survival gates.
                anchor_guard = bool(winner_reproduced or winner_anchor.exists())
                if current_skill_cutover >= args.skill_cutover_start:
                    # Shadow-only stage: macro remains authoritative, so skill
                    # execution records are intentionally zero. Promote only
                    # after the skill head can imitate non-KEEP macro work.
                    skill_eval_safe = (
                        anchor_guard
                        and shadow_ready
                        and eval_m["mean_margin"] >= best_margin - 2000.0
                        and eval_m["animal_escapes"] <= int(args.skill_max_animal_escapes)
                        and eval_m["plant_deaths"] / max(1, eval_m["games"]) <= float(args.skill_max_plant_deaths_per_game)
                    )
                else:
                    stage_ref_valid = (
                        skill_stage_reference_margin is not None
                        and skill_stage_reference_cutover is not None
                        and int(skill_stage_reference_cutover) == int(current_skill_cutover)
                    )
                    stage_margin_floor = (
                        float(skill_stage_reference_margin)
                        - float(args.skill_stage_margin_tolerance)
                        if stage_ref_valid
                        else best_margin - float(args.rollback_margin)
                    )
                    skill_eval_safe = (
                        anchor_guard
                        and shadow_ready
                        and eval_m["skill_records"] > 0
                        and eval_m["skill_nonkeep_proposals"] >= 32
                        and eval_m["skill_executed_sampled"] >= int(args.skill_min_executed_samples)
                        and eval_m["mean_margin"] >= stage_margin_floor
                        and eval_m["animal_escapes"] <= int(args.skill_max_animal_escapes)
                        and eval_m["plant_deaths"] / max(1, eval_m["games"]) <= float(args.skill_max_plant_deaths_per_game)
                        and skill_active_ratio >= float(args.skill_min_takeover_ratio)
                    )
                skill_stage_failed = (
                    current_skill_cutover < args.skill_cutover_start
                    and not skill_eval_safe
                )
                rolled_back = False
                improved = (
                    not skill_stage_failed
                    and (
                        eval_m["win_rate"] > best_win_rate
                        or (
                            eval_m["win_rate"] == best_win_rate
                            and eval_m["mean_margin"] > best_margin
                        )
                    )
                )
                if improved:
                    best_win_rate = eval_m["win_rate"]
                    best_margin = eval_m["mean_margin"]
                    save_state(
                        best,
                        actor,
                        optimizer,
                        iteration,
                        best_margin,
                        best_win_rate,
                        config,
                        extra=winner_state_extra(),
                    )
                safe_eval = (
                    not skill_stage_failed
                    and winner_reproduced
                    and eval_m["win_rate"] >= best_win_rate
                    and eval_m["mean_margin"] >= best_margin - args.rollback_margin
                )
                if safe_eval:
                    save_state(
                        safe_best,
                        actor,
                        optimizer,
                        iteration,
                        best_margin,
                        best_win_rate,
                        config,
                        extra=winner_state_extra(),
                    )
                    print(
                        f"[SAFE-BEST] iter={iteration:05d} "
                        f"eval_margin={eval_m['mean_margin']:+.1f} -> {safe_best.name}",
                        flush=True,
                    )
                if skill_eval_safe:
                    if (
                        skill_stage_reference_cutover is None
                        or int(skill_stage_reference_cutover) != int(current_skill_cutover)
                    ):
                        skill_stage_reference_margin = float(eval_m["mean_margin"])
                        skill_stage_reference_cutover = int(current_skill_cutover)
                    else:
                        skill_stage_reference_margin = max(
                            float(skill_stage_reference_margin),
                            float(eval_m["mean_margin"]),
                        )
                    save_state(
                        skill_stage_safe,
                        actor,
                        optimizer,
                        iteration,
                        best_margin,
                        best_win_rate,
                        config,
                        extra=winner_state_extra(),
                    )
                    print(
                        f"[SKILL-STAGE-SAFE] iter={iteration:05d} "
                        f"cutover={current_skill_cutover} "
                        f"margin={eval_m['mean_margin']:+.1f} -> {skill_stage_safe.name}",
                        flush=True,
                    )
                print_metrics(
                    "EVAL ",
                    iteration,
                    eval_m,
                    best=best_margin,
                )
                if improved:
                    print(
                        f"[BEST]  iter={iteration:05d} "
                        f"win_rate={100*best_win_rate:.2f}% "
                        f"mean_margin={best_margin:+.1f} -> {best}",
                        flush=True,
                    )
                elif (
                    best.exists()
                    and (
                        skill_stage_failed
                        or (
                            iteration > migration_guard_until
                            and eval_m["win_rate"] <= best_win_rate
                            and eval_m["mean_margin"] < best_margin - args.rollback_margin
                        )
                    )
                ):
                    if skill_stage_failed:
                        failed_snapshot = out_dir / "winner_v45_skill_failed_last.pt"
                        save_snapshot(
                            actor,
                            failed_snapshot,
                            {
                                "iteration": int(iteration),
                                "cutover_step": int(current_skill_cutover),
                                "eval_mean_margin": float(eval_m["mean_margin"]),
                                "plant_deaths": int(eval_m["plant_deaths"]),
                                "animal_escapes": int(eval_m["animal_escapes"]),
                                "skill_samples": int(eval_m["skill_samples"]),
                                "skill_nonkeep_proposals": int(
                                    eval_m["skill_nonkeep_proposals"]
                                ),
                                "skill_executed_sampled": int(
                                    eval_m["skill_executed_sampled"]
                                ),
                                "skill_takeover_ratio": float(skill_active_ratio),
                            },
                        )
                        print(
                            f"[SKILL-FAIL-SNAPSHOT] iter={iteration:05d} "
                            f"cutover={current_skill_cutover} -> {failed_snapshot.name}",
                            flush=True,
                        )
                    if skill_stage_failed and skill_stage_safe.exists():
                        rollback_path = skill_stage_safe
                    elif safe_best.exists():
                        rollback_path = safe_best
                    elif winner_reproduced and winner_anchor.exists():
                        rollback_path = winner_anchor
                    else:
                        rollback_path = best
                    rollback = torch.load(
                        rollback_path, map_location=device, weights_only=False
                    )
                    # Reset migration-only parameters first so rollback to an
                    # older V4.2 checkpoint also resets the V4.3 shop strategy.
                    init_actor(actor)
                    load_actor_state_compatible(
                        actor, rollback["actor_state"]
                    )
                    try:
                        optimizer.load_state_dict(rollback["optimizer_state"])
                    except Exception:
                        optimizer.state.clear()
                    if isinstance(rollback.get("micro_optimizer_state"), dict):
                        try:
                            micro_optimizer.load_state_dict(
                                rollback["micro_optimizer_state"]
                            )
                        except Exception:
                            micro_optimizer.state.clear()
                    else:
                        micro_optimizer.state.clear()
                    # Preserve conservative learning rates after rollback.
                    for group in micro_optimizer.param_groups:
                        group["lr"] = float(args.micro_learning_rate)
                    if isinstance(rollback.get("harvest_optimizer_state"), dict):
                        try:
                            harvest_optimizer.load_state_dict(rollback["harvest_optimizer_state"])
                        except Exception:
                            harvest_optimizer.state.clear()
                    else:
                        harvest_optimizer.state.clear()
                    for group in harvest_optimizer.param_groups:
                        group["lr"] = float(args.harvest_learning_rate)
                    if isinstance(rollback.get("winner_optimizer_state"), dict):
                        try:
                            winner_optimizer.load_state_dict(
                                rollback["winner_optimizer_state"]
                            )
                        except Exception:
                            winner_optimizer.state.clear()
                    skill_stage_reference_margin = rollback.get(
                        "skill_stage_reference_margin", skill_stage_reference_margin
                    )
                    skill_stage_reference_cutover = rollback.get(
                        "skill_stage_reference_cutover", skill_stage_reference_cutover
                    )
                    winner_reproduced = bool(
                        rollback.get("winner_reproduced", winner_reproduced)
                    )
                    current_skill_cutover = int(
                        rollback.get("skill_cutover_step", args.skill_cutover_start)
                    )
                    current_skill_cutover = max(
                        int(args.skill_cutover_min),
                        min(int(args.skill_cutover_start), current_skill_cutover),
                    )
                    skill_stable_evals = int(
                        rollback.get("skill_stable_evals", 0)
                    )
                    if skill_stage_failed:
                        skill_stable_evals = 0
                    rolled_back = True
                    actor.eval()
                    rollback_tag = "SKILL-ROLLBACK" if skill_stage_failed else "ROLLBACK"
                    print(
                        f"[{rollback_tag}] iter={iteration:05d} "
                        f"eval_margin={eval_m['mean_margin']:+.1f} "
                        f"death/game={eval_m['plant_deaths']/max(1,eval_m['games']):.1f} "
                        f"escapes={eval_m['animal_escapes']} "
                        f"restored={rollback_path.name} "
                        f"cutover={current_skill_cutover}",
                        flush=True,
                    )

                if not rolled_back and current_skill_cutover > args.skill_cutover_min:
                    if skill_eval_safe:
                        skill_stable_evals += 1
                    else:
                        skill_stable_evals = 0
                    print(
                        f"[SKILL-EVAL] iter={iteration:05d} "
                        f"cutover={current_skill_cutover} safe={skill_eval_safe} "
                        f"margin={eval_m['mean_margin']:+.1f} "
                        f"death/game={eval_m['plant_deaths']/max(1,eval_m['games']):.1f} "
                        f"escapes={eval_m['animal_escapes']} "
                        f"proposals={eval_m['skill_nonkeep_proposals']} "
                        f"executed={eval_m['skill_executed_sampled']} "
                        f"takeover={skill_active_ratio:.3f} "
                        f"anchor_guard={int(anchor_guard)} "
                        f"stage_ref={skill_stage_reference_margin if skill_stage_reference_margin is not None else 'NA'} "
                        f"stable={skill_stable_evals}/{args.skill_stable_evals_required}",
                        flush=True,
                    )
                    if skill_stable_evals >= args.skill_stable_evals_required:
                        old_cutover = int(current_skill_cutover)
                        current_skill_cutover = max(
                            int(args.skill_cutover_min),
                            old_cutover - int(args.skill_cutover_step_size),
                        )
                        skill_stable_evals = 0
                        print(
                            f"[SKILL-CURRICULUM] iter={iteration:05d} "
                            f"cutover {old_cutover}->{current_skill_cutover}",
                            flush=True,
                        )

            save_state(
                latest,
                actor,
                optimizer,
                iteration,
                best_margin,
                best_win_rate,
                config,
                extra=winner_state_extra(),
            )
            row = {
                "iteration": iteration,
                "temperature": temperature,
                "train_win_rate": train_m["win_rate"],
                "train_own_money": train_m["own_money"],
                "train_v51_money": train_m["v51_money"],
                "train_mean_margin": train_m["mean_margin"],
                "train_median_margin": train_m["median_margin"],
                "train_max_margin": train_m["max_margin"],
                "train_min_margin": train_m["min_margin"],
                "train_decisions": train_m["decisions"],
                "policy_loss": ppo["policy_loss"],
                "value_loss": ppo["value_loss"],
                "entropy": ppo["entropy"],
                "approx_kl": ppo["approx_kl"],
                "clip_fraction": ppo["clip_fraction"],
                "elite_games": elite_s["elite_games"],
                "elite_wins": elite_s["elite_wins"],
                "elite_best_margin": elite_s["elite_best_margin"],
                "elite_loss": elite_fit["elite_loss"],
                "elite_route_acc": elite_fit["elite_route_acc"],
                "winner_count": winner_s["winner_count"],
                "near_win_count": winner_s["near_win_count"],
                "good_count": winner_s["good_count"],
                "winner_bc_route_acc": winner_fit["route_acc"],
                "winner_bc_market_acc": winner_fit["market_acc"],
                "winner_bc_horizon_acc": winner_fit["horizon_acc"],
                "winner_reproduced": int(winner_reproduced),
                "verified_winner_margin": (
                    verified_winner_margin
                    if verified_winner_margin is not None else ""
                ),
                "eval_win_rate": (
                    eval_m["win_rate"] if evaluated else ""
                ),
                "eval_own_money": (
                    eval_m["own_money"] if evaluated else ""
                ),
                "eval_v51_money": (
                    eval_m["v51_money"] if evaluated else ""
                ),
                "eval_mean_margin": (
                    eval_m["mean_margin"] if evaluated else ""
                ),
                "best_win_rate": best_win_rate,
                "best_mean_margin": best_margin,
            }
            append_csv(csv_path, row)

            if evaluated and eval_m["win_rate"] >= 1.0:
                print(
                    "[TARGET] Current held-out evaluation reached 100% wins. "
                    "Training continues until Ctrl+C as requested.",
                    flush=True,
                )

    except KeyboardInterrupt:
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except Exception:
            pass
        print("\nCTRL+C received - saving current state...", flush=True)
    finally:
        if rollout_pool is not None:
            try:
                rollout_pool.terminate()
                rollout_pool.join()
            except Exception:
                pass
        save_state(
            latest,
            actor,
            optimizer,
            iteration,
            best_margin,
            best_win_rate,
            config,
            extra=winner_state_extra(),
        )
        save_snapshot(
            actor,
            out_dir / "winner_v4_actor_last.pt",
            {"iteration": iteration},
        )
        elite_games_final = []
        if elite_path.exists():
            try:
                elite_games_final = torch.load(
                    elite_path, map_location="cpu", weights_only=False
                ).get("games", [])
            except Exception:
                elite_games_final = []
        summary = {
            "iteration": iteration,
            "best_win_rate": best_win_rate,
            "best_mean_margin": (
                best_margin if math.isfinite(best_margin) else None
            ),
            **elite_summary(elite_games_final),
            **winner_summary(elite_games_final),
            "winner_reproduced": bool(winner_reproduced),
            "verified_winner_margin": verified_winner_margin,
            "verified_winner_seed": verified_winner_seed,
            "verified_winner_seat": verified_winner_seat,
            "latest": str(latest),
            "best": str(best),
            "winner_anchor": str(winner_anchor),
            "safe_best": str(safe_best),
            "elite_memory": str(elite_path),
            "metrics_csv": str(csv_path),
            "stopped_by_user_or_limit": True,
        }
        (out_dir / "winner_v4_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
