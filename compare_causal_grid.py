from pathlib import Path
import json
import torch

from winner_train import exact_games, metrics, partition_elite_games, run_actor_reproduction

ROOT = Path(__file__).resolve().parent

ACTORS = [
    ("BASE", ROOT / "output" / "winner_v4_latest.pt"),
    ("CAUSAL1", ROOT / "output" / "causal_v4_1_actor.pt"),
    ("TAU200", ROOT / "output" / "causal_all3_tau200_v4_1_actor.pt"),
    ("TAU400", ROOT / "output" / "causal_all3_tau400_v4_1_actor.pt"),
    ("TAU800", ROOT / "output" / "causal_all3_tau800_v4_1_actor.pt"),
]


def main():
    games = torch.load(
        ROOT / "output" / "winner_memory_v4.pt",
        map_location="cpu",
        weights_only=False,
    )["games"]
    wins = partition_elite_games(games)["winner"]
    results = {}
    for name, snap in ACTORS:
        print(f"=== {name} ===", flush=True)
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
            item = {
                "seed": int(game["seed"]),
                "seat": int(game["seat"]),
                "stored_margin": int(game["margin"]),
                "actor_margin": int(row["margin"]),
                "win": int(row["win"]),
            }
            winner_rows.append(item)
            print("WINNER", item, flush=True)

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
        winner_wins = sum(x["win"] for x in winner_rows)
        winner_mean = sum(x["actor_margin"] for x in winner_rows) / len(winner_rows)
        print(
            f"SCORE {name} winner_wins={winner_wins}/3 "
            f"winner_mean={winner_mean:+.1f} "
            f"heldout_wins={heldout['wins']}/8 "
            f"heldout_margin={heldout['mean_margin']:+.1f}",
            flush=True,
        )
        results[name] = {"winners": winner_rows, "heldout": heldout}

    out = ROOT / "output" / "causal_grid_compare_v4_1.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
