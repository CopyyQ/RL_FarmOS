import json
from collections import defaultdict
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent

def main():
    evidence = defaultdict(list)

    # Fresh exact route105 counterfactuals already include the base route.
    fresh = torch.load(
        ROOT / "output" / "route105_gate_dataset_v4_2.pt",
        map_location="cpu",
        weights_only=False,
    )
    for row in fresh["rows"]:
        evidence[int(row["base_route_id"])].append(
            {
                "source": "fresh16",
                "seed": int(row["seed"]),
                "seat": int(row["seat"]),
                "gain": int(row["gain"]),
            }
        )

    # Recover the step-144 base route of the three known winners from memory,
    # then attach the exact gate-vs-baseline gain from the A/B file.
    memory = torch.load(
        ROOT / "output" / "winner_memory_v4.pt",
        map_location="cpu",
        weights_only=False,
    )
    route_by_game = {}
    for game in memory["games"]:
        if int(game.get("margin", 0)) <= 0:
            continue
        route = None
        for record in game.get("records", []):
            if int(record["step"]) == 144:
                route = int(record.get("base_route_id", record["chosen_route_id"]))
                break
        if route is not None:
            route_by_game[(int(game["seed"]), int(game["seat"]))] = route

    ab = json.loads(
        (ROOT / "output" / "structural_gate_ab_v4_2.json").read_text()
    )
    baseline = {
        (int(x["seed"]), int(x["seat"])): int(x["margin"])
        for x in ab["baseline"]["known"]
    }
    gated = {
        (int(x["seed"]), int(x["seat"])): int(x["margin"])
        for x in ab["structural_gate"]["known"]
    }
    for key, base_margin in baseline.items():
        if key not in route_by_game or key not in gated:
            continue
        evidence[route_by_game[key]].append(
            {
                "source": "known3",
                "seed": key[0],
                "seat": key[1],
                "gain": int(gated[key] - base_margin),
            }
        )

    table = {}
    promoted = []
    for route, rows in sorted(evidence.items()):
        gains = [int(x["gain"]) for x in rows]
        stats = {
            "route_id": int(route),
            "n": len(gains),
            "mean_gain": sum(gains) / len(gains),
            "min_gain": min(gains),
            "max_gain": max(gains),
            "positive": sum(x > 0 for x in gains),
            "neutral": sum(x == 0 for x in gains),
            "negative": sum(x < 0 for x in gains),
            "evidence": rows,
        }
        stats["promoted"] = bool(
            stats["n"] >= 2
            and stats["mean_gain"] > 500.0
            and stats["negative"] == 0
        )
        if stats["promoted"]:
            promoted.append(int(route))
        table[str(route)] = stats

    payload = {
        "schema": "farmos_structural_gain_table_v4_2",
        "candidate_route": 105,
        "start_step": 144,
        "horizon": 48,
        "promotion_rule": {
            "min_samples": 2,
            "min_mean_gain": 500.0,
            "max_negative_samples": 0,
        },
        "source_routes": sorted(promoted),
        "routes": table,
    }

    out = ROOT / "output" / "structural_gain_table_v4_2.json"
    asset = ROOT / "assets" / "structural_gain_table_v4_2.json"
    text = json.dumps(payload, indent=2)
    out.write_text(text, encoding="utf-8")
    asset.write_text(text, encoding="utf-8")

    print("PROMOTED", sorted(promoted))
    for route, stats in table.items():
        print(
            route,
            "n", stats["n"],
            "mean", round(stats["mean_gain"], 1),
            "range", (stats["min_gain"], stats["max_gain"]),
            "neg", stats["negative"],
            "promoted", stats["promoted"],
        )

if __name__ == "__main__":
    main()
