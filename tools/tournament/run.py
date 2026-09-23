from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src"), str(ROOT)]

import winner_train as wt


DEFAULT_BASE = {
    "temperature": 1.00242003614173,
    "residual_scale": 1.0,
    "base_keep_bias": 2.3,
    "decision_every": 24,
    "skill_cutover_step": 672,
    "skill_keep_penalty": 2.75,
    "skill_shadow_start_step": 0,
    "skill_confidence_threshold": 0.70,
    "runtime_options": {
        "act_keep_gate_path": str(
            ROOT / "assets" / "act_keep_gate_sweep_best.pt"
        ),
        "act_keep_threshold": 0.50,
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def candidate_hash(candidate: dict[str, Any]) -> str:
    payload = {
        "rollout": candidate.get("rollout", {}),
        "runtime_options": candidate.get("runtime_options", {}),
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def compact_game(row: dict[str, Any]) -> dict[str, Any]:
    dense = dict(row.get("dense_stats") or {})
    portfolio = dict(row.get("portfolio_switch_stats") or {})
    event_counts = {}
    for event in list(row.get("portfolio_switch_events") or []):
        key = (
            f"{event.get('kind', '')}:"
            f"{event.get('from', '')}->{event.get('to', '')}:"
            f"b{int(event.get('step', 0) or 0) // 72}"
        )
        event_counts[key] = event_counts.get(key, 0) + 1
    return {
        "seed": int(row["seed"]),
        "seat": int(row["seat"]),
        "own_money": int(row["own_money"]),
        "v51_money": int(row["v51_money"]),
        "margin": int(row["margin"]),
        "win": int(row["win"]),
        "plant_deaths": int(dense.get("plant_deaths", 0) or 0),
        "animal_escapes": int(dense.get("animal_escapes", 0) or 0),
        "invalid_ops": int(dense.get("invalid_ops", 0) or 0),
        "harvested_units": int(dense.get("harvested_units", 0) or 0),
        "sold_units": int(dense.get("sold_units", 0) or 0),
        "portfolio_applied_steps": int(
            portfolio.get("applied_steps", 0) or 0
        ),
        "portfolio_seed_switches": int(
            portfolio.get("seed_switches", 0) or 0
        ),
        "portfolio_plant_switches": int(
            portfolio.get("plant_switches", 0) or 0
        ),
        "portfolio_event_counts": event_counts,
    }


def percentile_linear(values: list[float], q: float) -> float:
    xs = sorted(float(x) for x in values)
    if not xs:
        return 0.0
    pos = (len(xs) - 1) * max(0.0, min(1.0, q))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "games": 0,
            "wins": 0,
            "win_rate": 0.0,
            "mean_margin": 0.0,
            "median_margin": 0.0,
            "p10_margin": 0.0,
            "cvar10_margin": 0.0,
            "worst_margin": 0.0,
            "below_20k": 0,
            "below_30k": 0,
            "below_50k": 0,
            "plant_deaths": 0,
            "animal_escapes": 0,
            "invalid_ops": 0,
            "harvested_units": 0,
            "sold_units": 0,
            "portfolio_applied_steps": 0,
            "portfolio_seed_switches": 0,
            "portfolio_plant_switches": 0,
        }
    margins = [float(row["margin"]) for row in rows]
    ordered = sorted(margins)
    tail_n = max(1, int(math.ceil(0.10 * len(ordered))))
    games = len(rows)
    return {
        "games": games,
        "wins": sum(int(row["win"]) for row in rows),
        "win_rate": sum(int(row["win"]) for row in rows) / games,
        "mean_margin": statistics.fmean(margins),
        "median_margin": statistics.median(margins),
        "p10_margin": percentile_linear(margins, 0.10),
        "cvar10_margin": statistics.fmean(ordered[:tail_n]),
        "worst_margin": min(margins),
        "below_20k": sum(x <= -20000 for x in margins),
        "below_30k": sum(x <= -30000 for x in margins),
        "below_50k": sum(x <= -50000 for x in margins),
        "plant_deaths": sum(int(row["plant_deaths"]) for row in rows),
        "animal_escapes": sum(int(row["animal_escapes"]) for row in rows),
        "invalid_ops": sum(int(row["invalid_ops"]) for row in rows),
        "harvested_units": sum(int(row["harvested_units"]) for row in rows),
        "sold_units": sum(int(row["sold_units"]) for row in rows),
        "portfolio_applied_steps": sum(
            int(row.get("portfolio_applied_steps", 0) or 0)
            for row in rows
        ),
        "portfolio_seed_switches": sum(
            int(row.get("portfolio_seed_switches", 0) or 0)
            for row in rows
        ),
        "portfolio_plant_switches": sum(
            int(row.get("portfolio_plant_switches", 0) or 0)
            for row in rows
        ),
    }


def behavior_signature(rows: list[dict[str, Any]]) -> str:
    behavior = []
    for row in sorted(
        rows, key=lambda r: (int(r["seed"]), int(r["seat"]))
    ):
        behavior.append(
            {
                "seed": int(row["seed"]),
                "seat": int(row["seat"]),
                "events": dict(
                    sorted(
                        (row.get("portfolio_event_counts") or {}).items()
                    )
                ),
            }
        )
    return hashlib.sha256(
        canonical_json(behavior).encode()
    ).hexdigest()[:16]


def paired_deltas(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    base = {(int(r["seed"]), int(r["seat"])): r for r in baseline_rows}
    cand = {(int(r["seed"]), int(r["seat"])): r for r in candidate_rows}
    keys = sorted(set(base) & set(cand))
    deltas = [int(cand[k]["margin"]) - int(base[k]["margin"]) for k in keys]
    if not deltas:
        return {
            "games": 0,
            "mean_delta": 0.0,
            "median_delta": 0.0,
            "improved": 0,
            "equal": 0,
            "worse": 0,
            "best_delta": 0,
            "worst_delta": 0,
        }
    return {
        "games": len(deltas),
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "improved": sum(x > 0 for x in deltas),
        "equal": sum(x == 0 for x in deltas),
        "worse": sum(x < 0 for x in deltas),
        "best_delta": max(deltas),
        "worst_delta": min(deltas),
    }


def score_candidate(
    baseline_metrics: dict[str, Any],
    metrics: dict[str, Any],
    paired: dict[str, Any],
    stage: dict[str, Any],
) -> tuple[float, list[str]]:
    delta_mean = float(metrics["mean_margin"]) - float(
        baseline_metrics["mean_margin"]
    )
    delta_cvar = float(metrics["cvar10_margin"]) - float(
        baseline_metrics["cvar10_margin"]
    )
    delta_p10 = float(metrics["p10_margin"]) - float(
        baseline_metrics["p10_margin"]
    )
    delta_win = float(metrics["win_rate"]) - float(
        baseline_metrics["win_rate"]
    )
    score = (
        delta_mean
        + float(stage.get("cvar_weight", 0.35)) * delta_cvar
        + float(stage.get("p10_weight", 0.15)) * delta_p10
        + float(stage.get("win_rate_weight", 4000.0)) * delta_win
    )
    reasons = []
    if delta_cvar < float(stage.get("hard_cvar_floor", -3000.0)):
        reasons.append(f"cvar_delta={delta_cvar:.1f}")
    worst_delta = float(metrics["worst_margin"]) - float(
        baseline_metrics["worst_margin"]
    )
    if worst_delta < float(stage.get("hard_worst_floor", -7000.0)):
        reasons.append(f"worst_delta={worst_delta:.1f}")
    if int(metrics["animal_escapes"]) > int(
        baseline_metrics["animal_escapes"]
    ):
        reasons.append("animal_escapes_increased")
    if paired["games"] and float(paired["mean_delta"]) < float(
        stage.get("hard_mean_floor", -2000.0)
    ):
        reasons.append(
            f"paired_mean_delta={float(paired['mean_delta']):.1f}"
        )
    return float(score), reasons


class ResultCache:
    def __init__(self, root: Path, run_key: str):
        self.root = root / run_key
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, candidate_key: str) -> Path:
        return self.root / f"{candidate_key}.jsonl"

    def load(self, candidate_key: str) -> dict[tuple[int, int], dict[str, Any]]:
        path = self._path(candidate_key)
        rows: dict[tuple[int, int], dict[str, Any]] = {}
        if not path.exists():
            return rows
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows[(int(row["seed"]), int(row["seat"]))] = row
        return rows

    def append(self, candidate_key: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path = self._path(candidate_key)
        current = self.load(candidate_key)
        for row in rows:
            current[(int(row["seed"]), int(row["seat"]))] = row
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for key in sorted(current):
                handle.write(canonical_json(current[key]) + "\n")
        tmp.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_candidate(
    candidate: dict[str, Any],
    base: dict[str, Any],
) -> dict[str, Any]:
    rollout = dict(base)
    rollout.update(candidate.get("rollout", {}))
    runtime = deep_merge(
        dict(base.get("runtime_options", {})),
        candidate.get("runtime_options", {}),
    )
    rollout["runtime_options"] = runtime
    return rollout


def expected_keys(seeds: list[int]) -> set[tuple[int, int]]:
    return {(int(seed), seat) for seed in seeds for seat in (0, 1)}


def resolve_stage_seeds(stage: dict[str, Any]) -> list[int]:
    if "seeds" in stage:
        return [int(x) for x in stage["seeds"]]
    if "seed_start" in stage and "seed_count" in stage:
        start = int(stage["seed_start"])
        count = max(0, int(stage["seed_count"]))
        return list(range(start, start + count))
    raise ValueError(
        f"stage {stage.get('name', '?')} needs seeds or seed_start/seed_count"
    )


def evaluate_candidate(
    candidate: dict[str, Any],
    *,
    base: dict[str, Any],
    seeds: list[int],
    workers: int,
    parent: Path,
    snapshot: Path,
    opponent: Path,
    iteration: int,
    cache: ResultCache,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved = resolve_candidate(candidate, base)
    key = candidate_hash(candidate)
    cached = cache.load(key)
    wanted = expected_keys(seeds)
    missing = sorted(wanted - set(cached))

    missing_seeds = sorted({seed for seed, _ in missing})
    started = time.perf_counter()
    fresh_rows: list[dict[str, Any]] = []
    if missing_seeds:
        raw_rows = wt.exact_games(
            str(parent),
            str(snapshot),
            str(opponent),
            missing_seeds,
            min(max(1, int(workers)), max(1, len(missing_seeds) * 2)),
            True,
            float(resolved["temperature"]),
            float(resolved["residual_scale"]),
            float(resolved["base_keep_bias"]),
            int(resolved["decision_every"]),
            int(iteration),
            both_seats=True,
            skill_cutover_step=int(resolved["skill_cutover_step"]),
            skill_keep_penalty=float(resolved["skill_keep_penalty"]),
            skill_shadow_start_step=int(
                resolved["skill_shadow_start_step"]
            ),
            skill_confidence_threshold=float(
                resolved["skill_confidence_threshold"]
            ),
            runtime_options=dict(resolved["runtime_options"]),
        )
        fresh_rows = [compact_game(row) for row in raw_rows]
        cache.append(key, fresh_rows)
        cached = cache.load(key)

    rows = [cached[key_] for key_ in sorted(wanted)]
    return rows, {
        "candidate_key": key,
        "fresh_games": len(fresh_rows),
        "cached_games": len(rows) - len(fresh_rows),
        "elapsed_sec": time.perf_counter() - started,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidates",
        default=str(ROOT / "configs" / "tournament" / "runtime100.json"),
    )
    ap.add_argument(
        "--stages",
        default=str(ROOT / "configs" / "tournament" / "stages.json"),
    )
    ap.add_argument(
        "--snapshot",
        default=str(ROOT / "assets" / "tournament_actor_iter1205.pt"),
    )
    ap.add_argument(
        "--parent",
        default=str(ROOT / "assets" / "parent_promoted_v2.pt"),
    )
    ap.add_argument(
        "--opponent",
        default=str(ROOT / "assets" / "v51_main.py"),
    )
    ap.add_argument("--iteration", type=int, default=1205)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--stage", default="")
    ap.add_argument("--limit-candidates", type=int, default=0)
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "runs" / "candidate_tournament"),
    )
    args = ap.parse_args()

    candidates_payload = load_json(Path(args.candidates))
    candidates = list(candidates_payload["candidates"])
    base = deep_merge(DEFAULT_BASE, candidates_payload.get("base", {}))
    stages_payload = load_json(Path(args.stages))
    stages = list(stages_payload["stages"])

    if args.stage:
        stages = [stage for stage in stages if stage["name"] == args.stage]
        if not stages:
            raise SystemExit(f"unknown stage: {args.stage}")

    parent = Path(args.parent).resolve()
    snapshot = Path(args.snapshot).resolve()
    opponent = Path(args.opponent).resolve()
    for path in (parent, snapshot, opponent):
        if not path.exists():
            raise FileNotFoundError(path)

    workers = int(args.workers)
    if workers <= 0:
        workers = max(1, min(16, (os.cpu_count() or 4) - 4))

    source_files = (
        ROOT / "winner_train.py",
        ROOT / "v45_skill_runtime.py",
        ROOT / "v45_skill_runtime_actkeep.py",
        ROOT / "src" / "kaggrl" / "v4_farm_supervisor.py",
        ROOT / "src" / "kaggrl" / "v45_economics.py",
        ROOT / "tools" / "tournament" / "run.py",
    )
    source_fingerprint = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in source_files
    }
    run_identity = {
        "parent_sha256": sha256_file(parent),
        "snapshot_sha256": sha256_file(snapshot),
        "opponent_sha256": sha256_file(opponent),
        "source_fingerprint": source_fingerprint,
        "iteration": int(args.iteration),
        "stochastic": True,
        "base_config": base,
    }
    run_key = hashlib.sha256(
        canonical_json(run_identity).encode()
    ).hexdigest()[:16]
    output_dir = Path(args.output_dir).resolve()
    cache = ResultCache(output_dir / "cache", run_key)

    baseline = next(
        (candidate for candidate in candidates if candidate.get("baseline")),
        None,
    )
    if baseline is None:
        raise RuntimeError("candidate manifest must contain one baseline")

    survivors = list(candidates)
    if args.limit_candidates > 0:
        survivors = survivors[: int(args.limit_candidates)]
        if baseline not in survivors:
            survivors.insert(0, baseline)

    all_reports = {}
    print(
        f"[TOURNAMENT] run={run_key} workers={workers} "
        f"candidates={len(survivors)}",
        flush=True,
    )

    for stage_index, stage in enumerate(stages, 1):
        seeds = resolve_stage_seeds(stage)
        if baseline not in survivors:
            survivors.insert(0, baseline)
        stage_candidates = list(dict((c["id"], c) for c in survivors).values())
        print(
            f"[STAGE] {stage['name']} index={stage_index} "
            f"candidates={len(stage_candidates)} seeds={len(seeds)} "
            f"games_each={len(seeds)*2}",
            flush=True,
        )

        baseline_rows, baseline_cache = evaluate_candidate(
            baseline,
            base=base,
            seeds=seeds,
            workers=workers,
            parent=parent,
            snapshot=snapshot,
            opponent=opponent,
            iteration=args.iteration,
            cache=cache,
        )
        baseline_metrics = summarize(baseline_rows)

        rankings = []
        for index, candidate in enumerate(stage_candidates, 1):
            rows, cache_stats = evaluate_candidate(
                candidate,
                base=base,
                seeds=seeds,
                workers=workers,
                parent=parent,
                snapshot=snapshot,
                opponent=opponent,
                iteration=args.iteration,
                cache=cache,
            )
            metrics = summarize(rows)
            paired = paired_deltas(baseline_rows, rows)
            score, hard_fail = score_candidate(
                baseline_metrics, metrics, paired, stage
            )
            record = {
                "id": candidate["id"],
                "baseline": bool(candidate.get("baseline")),
                "candidate_key": candidate_hash(candidate),
                "behavior_signature": behavior_signature(rows),
                "score": score,
                "hard_fail": hard_fail,
                "metrics": metrics,
                "paired": paired,
                "cache": cache_stats,
                "candidate": candidate,
            }
            rankings.append(record)
            print(
                f"[{stage['name']}] {index:03d}/{len(stage_candidates):03d} "
                f"{candidate['id']} score={score:+.1f} "
                f"mean={metrics['mean_margin']:+.1f} "
                f"dmean={paired['mean_delta']:+.1f} "
                f"cvar={metrics['cvar10_margin']:+.1f} "
                f"fresh={cache_stats['fresh_games']} "
                f"fail={'|'.join(hard_fail) if hard_fail else '-'}",
                flush=True,
            )

        rankings.sort(
            key=lambda row: (
                bool(row["hard_fail"]),
                -float(row["score"]),
                -float(row["paired"]["mean_delta"]),
            )
        )
        keep = int(stage.get("keep", len(rankings)))
        eligible = [
            row for row in rankings
            if not row["hard_fail"] or bool(row["baseline"])
        ]
        if bool(stage.get("behavior_dedup", False)):
            selected_rows = []
            seen_signatures = set()
            for row in eligible:
                if bool(row["baseline"]):
                    continue
                signature = str(row.get("behavior_signature", ""))
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                selected_rows.append(row)
                if len(selected_rows) >= keep:
                    break
        else:
            selected_rows = eligible[:keep]
        if baseline["id"] not in {row["id"] for row in selected_rows}:
            selected_rows.append(
                next(row for row in rankings if row["baseline"])
            )
        survivor_ids = {row["id"] for row in selected_rows}
        survivors = [
            candidate
            for candidate in stage_candidates
            if candidate["id"] in survivor_ids
        ]

        report = {
            "stage": stage,
            "run_identity": run_identity,
            "workers": workers,
            "baseline_metrics": baseline_metrics,
            "baseline_cache": baseline_cache,
            "rankings": rankings,
            "selected": [row["id"] for row in selected_rows],
        }
        all_reports[stage["name"]] = report
        report_path = output_dir / f"{stage_index:02d}_{stage['name']}.json"
        write_json(report_path, report)
        print(
            f"[SELECT] {stage['name']} "
            f"kept={len(survivors)} "
            f"top={selected_rows[0]['id'] if selected_rows else 'none'} "
            f"report={report_path}",
            flush=True,
        )

    write_json(
        output_dir / "summary.json",
        {
            "run_identity": run_identity,
            "workers": workers,
            "survivors": [candidate["id"] for candidate in survivors],
            "reports": {
                name: {
                    "selected": report["selected"],
                    "baseline_metrics": report["baseline_metrics"],
                }
                for name, report in all_reports.items()
            },
        },
    )


if __name__ == "__main__":
    main()
