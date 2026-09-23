import json
import shutil
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"


def atomic_save(payload, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main():
    validation_path = OUT / "promotion_v4_2.json"
    if not validation_path.exists():
        raise FileNotFoundError(validation_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not bool(validation.get("passed")):
        raise RuntimeError("V4.2 promotion validation did not pass")

    latest = OUT / "winner_v4_latest.pt"
    best = OUT / "winner_v4_best.pt"
    reproduced = OUT / "winner_v4_reproduced.pt"
    causal = OUT / "causal_v4_1_actor.pt"

    for path in (latest, best, reproduced, causal):
        if not path.exists():
            raise FileNotFoundError(path)

    # Immutable backups of the pre-V4.2 state.
    for path in (latest, best, reproduced):
        backup = OUT / f"{path.stem}_pre_v4_2{path.suffix}"
        if not backup.exists():
            shutil.copy2(path, backup)

    base = torch.load(latest, map_location="cpu", weights_only=False)
    causal_payload = torch.load(causal, map_location="cpu", weights_only=False)
    actor_state = causal_payload.get("actor_state")
    if not isinstance(actor_state, dict) or len(actor_state) < 20:
        raise RuntimeError("causal actor_state is incomplete")

    config = dict(base.get("config") or {})
    config.update(
        {
            "run_mode": "winner_amplification_exact_v51_v4_2_causal_structural",
            "structural_gate": "exact_gain_table_v4_2",
            "structural_source_routes": validation.get("source_routes", []),
            "structural_gain_table_schema": "farmos_structural_gain_table_v4_2",
            "idle_weed_labor": True,
            "promoted_from": "causal_v4_1_actor.pt",
        }
    )

    fixed = validation["fixed_metrics"]
    fixed_win_rate = float(
        fixed.get(
            "win_rate",
            float(fixed.get("wins", 0)) / max(1, int(fixed.get("games", 1))),
        )
    )
    promoted = dict(base)
    promoted.update(
        {
            "schema": "farmos_v51_winner_amplification_state_v4_2",
            "actor_state": actor_state,
            # Optimizer moments were learned around the old 14-tensor actor.
            # Force a clean optimizer state while preserving model/memory.
            "optimizer_state": None,
            "winner_optimizer_state": None,
            "config": config,
            "best_margin": float(fixed["mean_margin"]),
            "best_win_rate": fixed_win_rate,
            "winner_reproduced": True,
            "winner_last_recheck_iteration": int(base.get("iteration", 82)),
            "v4_2_promotion": {
                "validation_file": str(validation_path),
                "known_metrics": validation["known_metrics"],
                "fixed_metrics": validation["fixed_metrics"],
                "fresh_metrics": validation["fresh_metrics"],
                "source_routes": validation.get("source_routes", []),
                "causal_actor": str(causal),
            },
        }
    )

    start = OUT / "winner_v4_2_start.pt"
    atomic_save(promoted, start)
    atomic_save(promoted, latest)
    atomic_save(promoted, best)
    atomic_save(promoted, reproduced)

    print("V4_2_PROMOTED")
    print("iteration", promoted.get("iteration"))
    print("actor_tensors", len(promoted["actor_state"]))
    print("best_margin", promoted["best_margin"])
    print("source_routes", validation.get("source_routes"))
    print("backups", [
        str(OUT / "winner_v4_latest_pre_v4_2.pt"),
        str(OUT / "winner_v4_best_pre_v4_2.pt"),
        str(OUT / "winner_v4_reproduced_pre_v4_2.pt"),
    ])


if __name__ == "__main__":
    main()
