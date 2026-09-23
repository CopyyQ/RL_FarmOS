from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


ACT_THRESHOLDS = (0.44, 0.47, 0.50, 0.53, 0.56)
SKILL_CONFIDENCE = (0.64, 0.68, 0.70, 0.74)
CUTOVER_STEPS = (624, 648, 672, 696, 720)

BASELINE = {
    "act_keep_threshold": 0.50,
    "skill_confidence_threshold": 0.70,
    "skill_cutover_step": 672,
}


def build_candidates():
    candidates = []
    for act, confidence, cutover in itertools.product(
        ACT_THRESHOLDS,
        SKILL_CONFIDENCE,
        CUTOVER_STEPS,
    ):
        baseline = (
            abs(act - BASELINE["act_keep_threshold"]) < 1e-12
            and abs(
                confidence - BASELINE["skill_confidence_threshold"]
            ) < 1e-12
            and cutover == BASELINE["skill_cutover_step"]
        )
        candidate = {
            "id": (
                f"act{act:.2f}_conf{confidence:.2f}_"
                f"cut{int(cutover)}"
            ),
            "baseline": baseline,
            "rollout": {
                "skill_confidence_threshold": float(confidence),
                "skill_cutover_step": int(cutover),
            },
            "runtime_options": {
                "act_keep_threshold": float(act),
            },
            "family": "runtime_gate_cutover",
        }
        candidates.append(candidate)

    candidates.sort(
        key=lambda row: (
            not bool(row.get("baseline")),
            row["id"],
        )
    )
    if len(candidates) != 100:
        raise RuntimeError(f"expected 100 candidates, got {len(candidates)}")
    if sum(bool(row.get("baseline")) for row in candidates) != 1:
        raise RuntimeError("expected exactly one baseline")
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default="configs/tournament/runtime100.json",
    )
    args = ap.parse_args()
    payload = {
        "schema": "farmos_candidate_manifest_v1",
        "description": (
            "100-candidate CPU tournament pilot over ACT/KEEP threshold, "
            "skill confidence threshold, and skill cutover. This validates "
            "the tournament infrastructure; future manifests can replace "
            "these runtime parameters with portfolio/market candidates."
        ),
        "base": {
            "temperature": 1.00242003614173,
            "residual_scale": 1.0,
            "base_keep_bias": 2.3,
            "decision_every": 24,
            "skill_keep_penalty": 2.75,
            "skill_shadow_start_step": 0,
        },
        "candidates": build_candidates(),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"WROTE {target} candidates={len(payload['candidates'])}")


if __name__ == "__main__":
    main()
