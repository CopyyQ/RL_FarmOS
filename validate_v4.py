from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from continuous_runtime import ContinuousRuntime
from rollout.v4_hybrid_agent import V4HybridRolloutAgent

ALLOWED_MARKETS = (
    "KEEP_ROUTE",
    "LIQUIDATE_SHED",
    "HOLD_SALES",
    "FRONT_RUN_1",
    "FRONT_RUN_9",
)


def parse_seeds(text: str):
    if ":" in text:
        start, count = [int(x) for x in text.split(":", 1)]
        return list(range(start, start + count))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def resolve_checkpoint(text: str) -> Path:
    if text:
        p = Path(text).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        return p
    for name in (
        "winner_v4_best.pt",
        "winner_v4_reproduced.pt",
        "winner_v4_latest.pt",
    ):
        p = ROOT / "output" / name
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "No V4 checkpoint found. Expected output/winner_v4_best.pt, "
        "winner_v4_reproduced.pt, or winner_v4_latest.pt"
    )


def preflight(checkpoint: Path):
    import torch
    import kaggle_environments
    from kaggle_environments import make

    engine_path = Path(kaggle_environments.__file__).resolve()
    vendor_root = (ROOT / "vendor").resolve()
    if vendor_root not in engine_path.parents:
        raise RuntimeError(
            f"validation must use bundled engine, got {engine_path}"
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload.get("actor_state"), dict):
        raise RuntimeError("checkpoint does not contain actor_state")

    # Construct the exact environment before spawning any workers.
    probe = make(
        "kaggriculture",
        configuration={"seed": 991337, "episodeSteps": 2},
        debug=False,
    )
    if len(probe.steps) != 1:
        raise RuntimeError("vendored Kaggriculture preflight failed")

    return {
        "engine_version": kaggle_environments.__version__,
        "engine_path": str(engine_path),
        "checkpoint_schema": payload.get("schema", "actor_snapshot"),
    }


def run_game(task):
    checkpoint, opponent, seed, seat, decision_every, base_keep_bias = task

    # Keep each validation worker lightweight.
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    from kaggle_environments import make

    runtime = ContinuousRuntime(
        str(ROOT / "assets" / "parent_promoted_v2.pt"),
        checkpoint,
        stochastic=False,
        temperature=0.90,
        residual_scale=1.0,
        base_keep_bias=base_keep_bias,
        decision_every=decision_every,
        allowed_market_modes=ALLOWED_MARKETS,
        seed=int(seed) * 101 + int(seat),
    )
    learner = V4HybridRolloutAgent(
        option_policy=runtime,
        min_option_confidence=0.0,
        enable_market_race_ordering=True,
    )
    agents = [learner, opponent] if seat == 0 else [opponent, learner]
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
            f"seed={seed} seat={seat}: invalid final statuses={statuses}"
        )
    farms = final[0].observation.farms
    own = int(farms[seat].money)
    rival = int(farms[1 - seat].money)
    margin = own - rival
    return {
        "seed": int(seed),
        "seat": int(seat),
        "own_money": own,
        "v51_money": rival,
        "margin": margin,
        "win": bool(margin > 0),
        "decisions": len(runtime.records),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="")
    ap.add_argument(
        "--opponent", default=str(ROOT / "assets" / "v51_main.py")
    )
    ap.add_argument("--seeds", default="23990000:10")
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 8) // 2)),
    )
    ap.add_argument("--decision-every", type=int, default=24)
    ap.add_argument("--base-keep-bias", type=float, default=2.3)
    ap.add_argument(
        "--output",
        default=str(ROOT / "output" / "validation_v4.json"),
    )
    args = ap.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    opponent = Path(args.opponent).resolve()
    if not opponent.exists():
        raise FileNotFoundError(f"v51 opponent not found: {opponent}")

    try:
        pf = preflight(checkpoint)
    except Exception as exc:
        print(f"[VALIDATION_PREFLIGHT_FAIL] {type(exc).__name__}: {exc}")
        raise SystemExit(3)

    print(
        f"[PREFLIGHT] PASS engine={pf['engine_version']} "
        f"checkpoint={checkpoint.name}",
        flush=True,
    )

    seeds = parse_seeds(args.seeds)
    tasks = [
        (
            str(checkpoint),
            str(opponent),
            seed,
            seat,
            args.decision_every,
            args.base_keep_bias,
        )
        for seed in seeds
        for seat in (0, 1)
    ]

    rows = []
    try:
        if args.workers <= 1:
            rows = [run_game(t) for t in tasks]
        else:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(processes=min(args.workers, len(tasks)))
            try:
                for i, row in enumerate(pool.imap_unordered(run_game, tasks), 1):
                    rows.append(row)
                    print(
                        f"[VALIDATE] {i:03d}/{len(tasks):03d} "
                        f"seed={row['seed']} seat={row['seat']} "
                        f"money={row['own_money']} v51={row['v51_money']} "
                        f"margin={row['margin']:+d}",
                        flush=True,
                    )
                pool.close()
                pool.join()
            except BaseException:
                pool.terminate()
                pool.join()
                raise
    except KeyboardInterrupt:
        print("[VALIDATION] Ctrl+C - stopped by user.", flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"[VALIDATION_RUNTIME_FAIL] {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise SystemExit(4)

    rows.sort(key=lambda x: (x["seed"], x["seat"]))
    margins = [r["margin"] for r in rows]
    own = [r["own_money"] for r in rows]
    rival = [r["v51_money"] for r in rows]
    wins = sum(int(r["win"]) for r in rows)
    summary = {
        "runtime_status": "PASS",
        "target_status": (
            "PASS_100_PERCENT" if wins == len(rows) else "NOT_YET_MET"
        ),
        "games": len(rows),
        "wins": wins,
        "losses": sum(int(r["margin"] < 0) for r in rows),
        "ties": sum(int(r["margin"] == 0) for r in rows),
        "win_rate": wins / max(1, len(rows)),
        "mean_own_money": statistics.fmean(own),
        "mean_v51_money": statistics.fmean(rival),
        "mean_margin": statistics.fmean(margins),
        "median_margin": statistics.median(margins),
        "min_margin": min(margins),
        "max_margin": max(margins),
    }
    report = {
        "preflight": pf,
        "checkpoint": str(checkpoint),
        "opponent": str(opponent),
        "seeds": seeds,
        "summary": summary,
        "games": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    print(
        "[VALIDATION] Runtime PASS. "
        + (
            "Hard target PASS."
            if summary["target_status"] == "PASS_100_PERCENT"
            else "Hard target not met yet; this is NOT a runtime error."
        ),
        flush=True,
    )
    # Deliberately exit 0 when exact games ran correctly, even if win-rate <100%.
    # Win-rate is a model-quality result, not an installation/runtime failure.


if __name__ == "__main__":
    mp.freeze_support()
    main()
