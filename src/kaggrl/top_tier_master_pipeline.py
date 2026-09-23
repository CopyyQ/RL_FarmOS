from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CapabilityStage:
    stage_id: str
    name: str
    purpose: str
    depends_on: tuple[str, ...]
    gate_artifact: str


STAGES: tuple[CapabilityStage, ...] = (
    CapabilityStage("S00", "contract_lock", "Freeze rules, action schema, observation schema, execution semantics, seeds, and reproducibility hashes.", (), "contract_gate.json"),
    CapabilityStage("S01", "teacher_demo_corpus", "Collect clean executable teacher demonstrations for mechanics only; no top-tier targets.", ("S00",), "teacher_corpus_gate.json"),
    CapabilityStage("S02", "gameplay_mechanics_base", "Train gameplay_base action-only with strategy context zeroed/frozen.", ("S01",), "gameplay_base_gate.json"),
    CapabilityStage("S03", "autonomous_gameplay_certification", "Run full games with teacher disabled; certify legality, money, inventory, clock, and no-crash behavior.", ("S02",), "autonomy_gate.json"),
    CapabilityStage("S04", "top_tier_corpus_quality", "Audit freshness, top-rank membership, diversity, full-action coverage, effective-action parity, and leakage.", ("S03",), "expert_corpus_gate.json"),
    CapabilityStage("S05", "expert_full_action_imitation", "Fine-tune gameplay_base only on top-tier trajectories across farmer, hands, and market actions.", ("S04",), "expert_bc_gate.json"),
    CapabilityStage("S06", "long_horizon_temporal_imitation", "Verify recurrent state carries expert plans across chunks and long macro horizons without route-teacher constraints.", ("S05",), "temporal_gate.json"),
    CapabilityStage("S07", "multi_expert_strategy_specialization", "Unfreeze strategy embeddings and learn distinct top-tier expert styles without collapsing to one mode.", ("S06",), "strategy_gate.json"),
    CapabilityStage("S08", "economic_resource_representation", "Learn cash, inventory, land, labor, production, and future-resource auxiliary targets from real top-tier outcomes.", ("S07",), "economy_gate.json"),
    CapabilityStage("S09", "return_value_calibration", "Train and calibrate value/return heads against terminal margin and intermediate economic outcomes.", ("S08",), "value_gate.json"),
    CapabilityStage("S10", "advantage_weighted_imitation", "Reweight expert decisions by estimated advantage/return while remaining inside expert support.", ("S09",), "awi_gate.json"),
    CapabilityStage("S11", "conservative_offline_rl", "Apply conservative offline policy improvement with explicit OOD/action-support safeguards.", ("S10",), "offline_rl_gate.json"),
    CapabilityStage("S12", "expert_state_counterfactuals", "Evaluate alternative actions from expert-visited states and retain only causally supported improvements.", ("S11",), "counterfactual_gate.json"),
    CapabilityStage("S13", "opponent_market_modeling", "Model opponent inventory, shared-market pressure, price/supply interaction, and strategic interference.", ("S12",), "opponent_market_gate.json"),
    CapabilityStage("S14", "population_self_play_league", "Train against a population of strong agents and historical snapshots rather than one fixed opponent.", ("S13",), "league_gate.json"),
    CapabilityStage("S15", "exploit_discovery", "Search for brittle states, adversarial openings, market races, resource starvation, and policy oscillation.", ("S14",), "exploit_gate.json"),
    CapabilityStage("S16", "hard_state_curriculum", "Feed discovered failures back as prioritized hard-state curriculum without replacing expert anchors.", ("S15",), "curriculum_gate.json"),
    CapabilityStage("S17", "self_play_rl_champion_challenger", "Run RL/self-play updates with champion-challenger promotion and rollback on regressions.", ("S16",), "selfplay_rl_gate.json"),
    CapabilityStage("S18", "cpu_distillation_runtime", "Distill/export the improved policy into Kaggle CPU constraints with Torch-to-NumPy/runtime parity.", ("S17",), "cpu_gate.json"),
    CapabilityStage("S19", "broad_closed_loop_stress", "Benchmark many unseen seeds, both seats, multiple strong opponents, tail-risk seeds, and repeated determinism.", ("S18",), "stress_gate.json"),
    CapabilityStage("S20", "submission_champion_gate", "Package only a candidate passing legality, parity, latency, robustness, and closed-loop promotion thresholds.", ("S19",), "submission_gate.json"),
)


def validate_stage_graph(stages: Iterable[CapabilityStage] = STAGES) -> None:
    stages = tuple(stages)
    ids = [stage.stage_id for stage in stages]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate stage ids")
    known: set[str] = set()
    for stage in stages:
        missing = [dep for dep in stage.depends_on if dep not in known]
        if missing:
            raise RuntimeError(
                f"stage {stage.stage_id} has unresolved dependencies: {missing}"
            )
        known.add(stage.stage_id)


def manifest() -> dict:
    validate_stage_graph()
    return {
        "pipeline": "top_tier_champion_v1",
        "principles": {
            "teacher_role": "mechanics_only",
            "top_tier_role": "strategy_and_skill_source_of_truth",
            "teacher_allowed_after_S03": False,
            "teacher_policy_target_allowed_in_expert_stage": False,
            "q_route_wrapper_is_primary_policy": False,
            "all_gates_required_for_champion": True,
        },
        "stage_count": len(STAGES),
        "stages": [asdict(stage) for stage in STAGES],
    }


def write_manifest(root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "pipeline_manifest.json"
    path.write_text(json.dumps(manifest(), indent=2, sort_keys=True) + "\n")
    for stage in STAGES:
        (root / "stages" / stage.stage_id).mkdir(parents=True, exist_ok=True)
    return path


def gate_path(root: Path, stage: CapabilityStage) -> Path:
    return Path(root) / "stages" / stage.stage_id / stage.gate_artifact


def stage_status(root: Path) -> list[dict]:
    validate_stage_graph()
    passed: set[str] = set()
    rows: list[dict] = []
    for stage in STAGES:
        path = gate_path(root, stage)
        payload = None
        gate_passed = False
        if path.is_file():
            payload = json.loads(path.read_text())
            gate_passed = bool(payload.get("passed", False))
        deps_passed = all(dep in passed for dep in stage.depends_on)
        effective_pass = gate_passed and deps_passed
        if effective_pass:
            passed.add(stage.stage_id)
        rows.append({
            "stage_id": stage.stage_id,
            "name": stage.name,
            "dependencies_passed": deps_passed,
            "gate_exists": path.is_file(),
            "gate_passed": gate_passed,
            "effective_pass": effective_pass,
            "gate": str(path),
        })
    return rows


def next_stage(root: Path) -> CapabilityStage | None:
    rows = {row["stage_id"]: row for row in stage_status(root)}
    for stage in STAGES:
        row = rows[stage.stage_id]
        if not row["effective_pass"] and row["dependencies_passed"]:
            return stage
    return None


def champion_ready(root: Path) -> bool:
    rows = stage_status(root)
    return bool(rows) and all(row["effective_pass"] for row in rows)
