from pathlib import Path
import json
import torch

from winner_train import (
    exact_games,
    metrics,
    partition_elite_games,
    run_actor_reproduction,
)

ROOT = Path(__file__).resolve().parent


def main():
    games = torch.load(
        ROOT / "output" / "winner_memory_v4.pt",
        map_location="cpu",
        weights_only=False,
    )["games"]
    wins = partition_elite_games(games)["winner"]
    actors = [
        ("BASE", ROOT / "output" / "winner_v4_latest.pt"),
        ("CAUSAL", ROOT / "output" / "causal_v4_1_actor.pt"),
    ]
    result = {}
    for name, snap in actors:
        print(f"=== {name} WINNERS ===", flush=True)
        winner_rows = []
        for game in wins:
            row = run_actor_reproduction(
                game,
                ROOT / "assets" / "parent_promoted_v2.pt",
                snap,
                ROOT / "assets" / "v51_main.py",
                temperature=0.90,
                residual_scale=1.0,
                base_keep_bias=2.3,
                decision_every=24,
                iteration=999,
            )
            winner_rows.append(
                {
                    "seed": int(game["seed"]),
                    "seat": int(game["seat"]),
                    "stored_margin": int(game["margin"]),
                    "actor_margin": int(row["margin"]),
                    "win": int(row["win"]),
                }
            )
            print(winner_rows[-1], flush=True)

        print(f"=== {name} HELDOUT ===", flush=True)
        rows = exact_games(
            ROOT / "assets" / "parent_promoted_v2.pt",
            snap,
            ROOT / "assets" / "v51_main.py",
            list(range(23990000, 23990004)),
            8,
            False,
            0.90,
            1.0,
            2.3,
            24,
            999,
            both_seats=True,
        )
        heldout = metrics(rows)
        print(json.dumps(heldout, indent=2), flush=True)
        result[name] = {
            "winners": winner_rows,
            "heldout": heldout,
        }

    out = ROOT / "output" / "causal_compare_v4_1.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
