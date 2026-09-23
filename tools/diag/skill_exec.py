from pathlib import Path
import os
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src"), str(ROOT)]

from winner_train import (
    exact_games, metrics, split_skill_teacher_rows,
    skill_teacher_records_to_batch, skill_bc_update,
)
from v45_skill_runtime import V45SkillRuntime
from kaggrl.v4_farm_supervisor import MICRO_TASKS

PARENT = ROOT / "assets/parent_promoted_v2.pt"
SNAP = Path(os.environ.get("FARMOS_CHECKPOINT", ROOT / "runs/v46_econ_shadow/winner_v45_skill_stage_safe.pt"))
OPP = ROOT / "assets/v51_main.py"

def main():
    seeds = list(range(26092301, 26092333))
    rows = exact_games(
        PARENT, SNAP, OPP, seeds,
        32, False, 1.0, 1.0, 2.3, 24, 1056,
        both_seats=False,
        skill_cutover_step=720,
        skill_keep_penalty=2.75,
        skill_shadow_start_step=0,
        skill_confidence_threshold=0.70,
    )
    m = metrics(rows)
    print("GAME", {
        "games": m["games"], "wins": m["wins"],
        "mean_margin": round(m["mean_margin"], 1),
        "median_margin": round(m["median_margin"], 1),
        "plant_deaths_per_game": round(m["plant_deaths"]/max(1,m["games"]), 2),
        "animal_escapes": m["animal_escapes"],
    }, flush=True)
    print("EXEC", {
        "critical_water_opportunities": m["critical_water_opportunities"],
        "water_attempts": m["water_attempts"],
        "water_success_actions": m["water_success_actions"],
        "water_success_rate": round(m["water_success_actions"]/max(1,m["water_attempts"]), 4),
        "fert_collect_opportunities": m["fert_collect_opportunities"],
        "fert_collect_attempts": m["fert_collect_attempts"],
        "fert_collect_success_actions": m["fert_collect_success_actions"],
        "fert_collect_success_rate": round(m["fert_collect_success_actions"]/max(1,m["fert_collect_attempts"]), 4),
        "fertilizer_units": m["fertilizer_collected"],
    }, flush=True)

    train_rows, val_rows = split_skill_teacher_rows(rows)
    val_records = [r for g in val_rows for r in (g.get("skill_teacher_records") or [])]
    print("SPLIT", {"train_games": len(train_rows), "val_games": len(val_rows), "val_records": len(val_records)}, flush=True)

    rt = V45SkillRuntime(
        str(PARENT), str(SNAP),
        stochastic=False,
        temperature=1.0,
        residual_scale=1.0,
        base_keep_bias=2.3,
        decision_every=24,
        allowed_market_modes=None,
        seed=123,
        skill_cutover_step=720,
        skill_keep_penalty=2.75,
        skill_shadow_start_step=0,
        skill_confidence_threshold=0.70,
    )
    actor = rt.base.actor
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    batch = skill_teacher_records_to_batch(
        [{"skill_teacher_records": val_records}],
        int(payload["hidden_dim"]), int(payload["clock_dim"]), torch.device("cpu")
    )
    val = skill_bc_update(
        actor, batch, None,
        temperature=1.0, epochs=0, minibatch=1024,
        keep_penalty=2.75, update=False,
    )
    print("VAL", {
        "rows": val["rows"],
        "acc": round(val["acc"],4),
        "nonkeep_acc": round(val["nonkeep_acc"],4),
        "core_min": round(val["core_min_acc"],4),
    }, flush=True)
    for i, name in enumerate(MICRO_TASKS):
        n = int(val["target_counts"][i])
        if n <= 0:
            continue
        p = val["per_task_precision"][i]
        r = val["per_task_recall"][i]
        f = val["per_task_f1"][i]
        print("TASK", name, "n", n, "P", round(float(p),4) if p==p else None,
              "R", round(float(r),4) if r==r else None,
              "F1", round(float(f),4) if f==f else None, flush=True)

if __name__ == "__main__":
    main()
