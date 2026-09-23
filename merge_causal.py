import argparse
import csv
import json
from pathlib import Path

from causal_refine import summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = []
    winners = []
    seen_winners = set()
    for name in args.inputs:
        payload = json.loads(Path(name).read_text(encoding="utf-8"))
        rows.extend(payload.get("rows", []))
        for winner in payload.get("winners", []):
            key = (int(winner["seed"]), int(winner["seat"]))
            if key not in seen_winners:
                seen_winners.add(key)
                winners.append(winner)

    summary = summarize(rows)
    out = {
        "schema": "farmos_v51_causal_refine_v4_1_merged",
        "winner_count": len(winners),
        "task_count": len(rows),
        "winners": winners,
        "summary": summary,
        "rows": rows,
    }
    path = Path(args.output)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if summary:
        with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    improved = [
        x for x in summary if int(x.get("best_gain_vs_original", 0)) > 0
    ]
    improved.sort(
        key=lambda x: int(x["best_gain_vs_original"]),
        reverse=True,
    )
    print(
        f"MERGED winners={len(winners)} rows={len(rows)} "
        f"groups={len(summary)} improved={len(improved)}"
    )
    for item in improved[:20]:
        print(
            f"{item['seed']} seat={item['seat']} {item['kind']} "
            f"step={item['step']} {item['original_choice']}->"
            f"{item['best_choice']} gain={item['best_gain_vs_original']:+d} "
            f"margin={item['best_margin']:+d}"
        )


if __name__ == "__main__":
    main()
