from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src"), str(ROOT)]

from kaggle_environments import make

from kaggrl.v4_farm_supervisor import apply_portfolio_switch_overlay
from kaggrl.v45_economics import CROP_META, ANIMAL_META
from rollout.v4_hybrid_agent import V4HybridRolloutAgent
from v45_skill_runtime_actkeep import V45SkillActKeepRuntime
from winner_train import ALLOWED_MARKETS


ITERATION = 1205
TEMPERATURE = 1.00242003614173
PARENT = str(ROOT / "assets" / "parent_promoted_v2.pt")
SNAPSHOT = str(ROOT / "assets" / "tournament_actor_iter1205.pt")
OPPONENT = str(ROOT / "assets" / "v51_main.py")
GATE = str(ROOT / "assets" / "act_keep_gate_sweep_best.pt")


def baseline_agent(seed, seat):
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
        enable_portfolio_switch=False,
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


def run_baseline(seed, seat):
    learner = baseline_agent(seed, seat)
    env = make(
        "kaggriculture",
        configuration={"seed": int(seed), "episodeSteps": 720},
        debug=False,
    )
    env.run(
        [learner, OPPONENT] if seat == 0 else [OPPONENT, learner]
    )
    return env


def relevant_action(action):
    action = action if isinstance(action, dict) else {}
    for order in list(action.get("market") or []):
        if not isinstance(order, (list, tuple)) or len(order) < 2:
            continue
        op = str(order[0])
        item = str(order[1])
        if op == "BUY_SEED" and item in CROP_META:
            return True
        if op == "BUY_ANIMAL" and item in ANIMAL_META:
            return True
    commands = [action.get("farmer")]
    commands.extend(list(action.get("hands") or []))
    for command in commands:
        if (
            isinstance(command, (list, tuple))
            and len(command) >= 2
            and str(command[0]) == "PLANT"
            and str(command[1]) in CROP_META
        ):
            return True
    return False


def candidate_kwargs(options):
    return {
        "crop_improvement_ratio": float(
            options.get("portfolio_crop_improvement_ratio", 1.30)
        ),
        "feed_reserve_days": float(
            options.get("portfolio_feed_reserve_days", 2.0)
        ),
        "activation_step": int(
            options.get("portfolio_activation_step", 144)
        ),
        "projected_horizon_extra_days": float(
            options.get("portfolio_horizon_extra_days", 2.0)
        ),
        "min_undersupply_ratio": float(
            options.get("portfolio_min_undersupply_ratio", 0.0)
        ),
        "min_shop_demand": float(
            options.get("portfolio_min_shop_demand", 1.0)
        ),
        "source_mode": str(
            options.get("portfolio_source_mode", "wheat")
        ),
        "objective": str(
            options.get("portfolio_objective", "best_crop_roi")
        ),
        "forced_animal": options.get("portfolio_forced_animal"),
        "animal_min_roi": float(
            options.get("portfolio_animal_min_roi", 1.0)
        ),
        "animal_payback_margin": float(
            options.get("portfolio_animal_payback_margin", 0.0)
        ),
        "diversity_penalty": float(
            options.get("portfolio_diversity_penalty", 0.20)
        ),
    }


def evaluate_candidate(candidate, trajectories, catastrophe_seed, normal_seed):
    options = dict(candidate.get("runtime_options") or {})
    if not bool(options.get("enable_portfolio_switch", False)):
        return {
            "id": candidate["id"],
            "baseline": True,
            "score": 0.0,
            "gate_pass": True,
            "requires_tomato_adaptation": False,
            "catastrophe_tomato_switches": 0,
            "catastrophe_wrong_crop_switches": 0,
            "normal_switches": 0,
            "normal_tomato_forces": 0,
            "normal_wool_breaks": 0,
            "normal_wool_support_switches": 0,
            "normal_crop_switches": 0,
            "normal_animal_switches": 0,
            "wrong_target_switches": 0,
            "early_switches": 0,
            "events": {},
            "behavior_signature": "baseline",
        }

    kwargs = candidate_kwargs(options)
    counts = Counter()
    cat_tomato = 0
    catastrophe_wrong_crop = 0
    normal_switches = 0
    normal_tomato_forces = 0
    normal_wool_breaks = 0
    normal_wool_support = 0
    normal_crop_switches = 0
    normal_animal_switches = 0
    wrong = 0
    early = 0

    for (seed, seat), env in trajectories.items():
        for step in range(len(env.steps) - 1):
            action = env.steps[step + 1][seat].action or {}
            if step < int(kwargs["activation_step"]):
                continue
            if not relevant_action(action):
                continue
            obs = env.steps[step][seat].observation
            out, meta = apply_portfolio_switch_overlay(
                obs, action, **kwargs
            )
            del out
            for event in list(meta.get("events") or []):
                kind = str(event.get("kind", ""))
                src = str(event.get("from", ""))
                dst = str(event.get("to", ""))
                key = f"{seed}:{seat}:{kind}:{src}->{dst}:d{step//24}"
                counts[key] += 1
                is_crop = kind in ("switch_seed", "switch_plant")
                is_animal = kind == "switch_animal"
                if seed == catastrophe_seed and is_crop:
                    if dst == "TOMATO":
                        cat_tomato += 1
                    else:
                        catastrophe_wrong_crop += 1
                        wrong += 1
                if seed == normal_seed:
                    normal_switches += 1
                    if is_crop:
                        normal_crop_switches += 1
                        if dst == "TOMATO":
                            normal_tomato_forces += 1
                    if is_animal:
                        normal_animal_switches += 1
                        if src == "SHEEP" and dst != "SHEEP":
                            normal_wool_breaks += 1
                        if src != "SHEEP" and dst == "SHEEP":
                            normal_wool_support += 1
                if step < 12 * 24:
                    early += 1

    behavior_payload = json.dumps(
        sorted(counts.items()), separators=(",", ":")
    )
    signature = hashlib.sha256(
        behavior_payload.encode()
    ).hexdigest()[:16]

    # Hard-gate semantics are regime-specific: crop-capable candidates must
    # recognize the catastrophic TOMATO opportunity, while the normal control
    # must never be forced into TOMATO and an existing SHEEP/WOOL path must not
    # be redirected away from SHEEP. Other animal switches are not blanket
    # failures because they may strengthen the WOOL regime.
    requires_tomato_adaptation = str(kwargs["objective"]) != "animal"
    gate_pass = (
        normal_tomato_forces == 0
        and normal_wool_breaks == 0
        and (not requires_tomato_adaptation or cat_tomato > 0)
    )
    score = (
        100.0 * cat_tomato
        + 30.0 * normal_wool_support
        - 250.0 * normal_tomato_forces
        - 300.0 * normal_wool_breaks
        - 120.0 * catastrophe_wrong_crop
        - 20.0 * early
    )
    return {
        "id": candidate["id"],
        "baseline": False,
        "score": float(score),
        "gate_pass": bool(gate_pass),
        "requires_tomato_adaptation": bool(requires_tomato_adaptation),
        "catastrophe_tomato_switches": int(cat_tomato),
        "catastrophe_wrong_crop_switches": int(catastrophe_wrong_crop),
        "normal_switches": int(normal_switches),
        "normal_tomato_forces": int(normal_tomato_forces),
        "normal_wool_breaks": int(normal_wool_breaks),
        "normal_wool_support_switches": int(normal_wool_support),
        "normal_crop_switches": int(normal_crop_switches),
        "normal_animal_switches": int(normal_animal_switches),
        "wrong_target_switches": int(wrong),
        "early_switches": int(early),
        "events": dict(sorted(counts.items())),
        "behavior_signature": signature,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidates",
        default=str(
            ROOT / "configs" / "tournament" / "economic100.json"
        ),
    )
    ap.add_argument("--catastrophe-seed", type=int, default=23238530)
    ap.add_argument("--normal-seed", type=int, default=23238531)
    ap.add_argument("--keep", type=int, default=20)
    ap.add_argument(
        "--output",
        default=str(ROOT / "runs" / "economic_shadow_screen.json"),
    )
    ap.add_argument(
        "--selected-manifest",
        default=str(
            ROOT
            / "runs"
            / "economic_shadow_top20_candidates.json"
        ),
    )
    args = ap.parse_args()

    manifest = json.loads(
        Path(args.candidates).read_text(encoding="utf-8")
    )
    candidates = list(manifest["candidates"])
    family_by_id = {
        candidate["id"]: str(candidate.get("family", "unclassified"))
        for candidate in candidates
    }

    trajectories = {}
    for seed in (args.catastrophe_seed, args.normal_seed):
        for seat in (0, 1):
            print(
                f"[SHADOW-BASELINE] seed={seed} seat={seat}",
                flush=True,
            )
            trajectories[(seed, seat)] = run_baseline(seed, seat)

    reports = []
    for index, candidate in enumerate(candidates, 1):
        report = evaluate_candidate(
            candidate,
            trajectories,
            args.catastrophe_seed,
            args.normal_seed,
        )
        reports.append(report)
        print(
            f"[SHADOW] {index:03d}/{len(candidates):03d} "
            f"{report['id']} score={report['score']:+.0f} "
            f"gate={int(bool(report.get('gate_pass', report['baseline'])))} "
            f"cat_tom={report['catastrophe_tomato_switches']} "
            f"normal_tom={report.get('normal_tomato_forces', 0)} "
            f"wool_break={report.get('normal_wool_breaks', 0)} "
            f"wool_support={report.get('normal_wool_support_switches', 0)} "
            f"early={report['early_switches']}",
            flush=True,
        )

    reports.sort(
        key=lambda row: (
            not bool(row.get("gate_pass", False)),
            bool(row["baseline"]),
            -float(row["score"]),
            -int(row["catastrophe_tomato_switches"]),
            int(row.get("normal_tomato_forces", 0)),
            int(row.get("normal_wool_breaks", 0)),
        )
    )
    selected = []
    selected_ids = set()
    seen = set()
    eligible = [
        row for row in reports
        if not row["baseline"] and bool(row.get("gate_pass", False))
    ]

    # Preserve at least one passing behavior from every surviving semantic
    # family before filling the remaining slots by score. This prevents the
    # hard gate from recreating the V1 failure mode where many configurations
    # collapse into one dominant crop-only behavior family.
    families = sorted({family_by_id.get(row["id"], "unclassified") for row in eligible})
    for family in families:
        for row in eligible:
            if family_by_id.get(row["id"], "unclassified") != family:
                continue
            # Semantic family preservation is stronger than shadow-signature
            # dedup here: a family may be dormant on the two gate seeds but
            # activate on the broader 12-game targeted stage.
            signature = row["behavior_signature"]
            seen.add(signature)
            selected.append(row["id"])
            selected_ids.add(row["id"])
            break
        if len(selected) >= int(args.keep):
            break

    for row in eligible:
        if len(selected) >= int(args.keep):
            break
        if row["id"] in selected_ids:
            continue
        signature = row["behavior_signature"]
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(row["id"])
        selected_ids.add(row["id"])

    baseline = next(
        candidate for candidate in candidates
        if candidate.get("baseline")
    )
    chosen_ids = set(selected)
    selected_candidates = [baseline] + [
        candidate
        for candidate in candidates
        if candidate["id"] in chosen_ids
    ]

    result = {
        "catastrophe_seed": args.catastrophe_seed,
        "normal_seed": args.normal_seed,
        "candidate_count": len(candidates),
        "selected": selected,
        "reports": reports,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")

    selected_manifest = dict(manifest)
    selected_manifest["description"] = (
        str(manifest.get("description", ""))
        + " Shadow-screen selected subset."
    )
    selected_manifest["candidates"] = selected_candidates
    selected_path = Path(args.selected_manifest)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text(
        json.dumps(selected_manifest, indent=2),
        encoding="utf-8",
    )
    print(
        f"[SHADOW-DONE] selected={len(selected)} "
        f"manifest={selected_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
