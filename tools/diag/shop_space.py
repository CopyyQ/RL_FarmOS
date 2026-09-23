from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations_with_replacement
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src"), str(ROOT)]

try:
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        MAX_SHOP_INSTANCES,
        SHOPS,
    )
    SHOP_NAMES = tuple(sorted(SHOPS))
except ModuleNotFoundError:
    # Diagnostics must remain runnable without the full Kaggle dependency
    # stack. These names are the same eight engine shops already consumed by
    # the economics module; MAX_SHOP_INSTANCES is fixed by the game rules.
    from kaggrl.v45_economics import ECON_SHOP_PRODUCTS

    SHOP_NAMES = tuple(sorted(ECON_SHOP_PRODUCTS))
    MAX_SHOP_INSTANCES = 8


def multinomial_probability(counts, draws, n_types):
    ways = math.factorial(draws)
    for count in counts.values():
        ways //= math.factorial(int(count))
    return ways / float(n_types ** draws)


def enumerate_shop_multisets(draws):
    rows = []
    for combo in combinations_with_replacement(SHOP_NAMES, int(draws)):
        counts = Counter(combo)
        probability = multinomial_probability(
            counts,
            int(draws),
            len(SHOP_NAMES),
        )
        rows.append(
            {
                "draws": int(draws),
                "shops": dict(sorted(counts.items())),
                "probability": probability,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default=str(ROOT / "runs" / "shop_space.json"),
    )
    args = ap.parse_args()

    stages = {}
    total_states = 0
    for draws in range(1, int(MAX_SHOP_INSTANCES) + 1):
        rows = enumerate_shop_multisets(draws)
        mass = sum(row["probability"] for row in rows)
        stages[str(draws)] = {
            "unique_multisets": len(rows),
            "probability_mass": mass,
            "states": rows,
        }
        total_states += len(rows)
        print(
            f"[SHOP-SPACE] draws={draws} "
            f"multisets={len(rows)} probability_mass={mass:.12f}",
            flush=True,
        )

    payload = {
        "shop_types": list(SHOP_NAMES),
        "shop_type_count": len(SHOP_NAMES),
        "max_shop_instances": int(MAX_SHOP_INSTANCES),
        "total_prefix_multisets": total_states,
        "ordered_sequences_final": len(SHOP_NAMES)
        ** int(MAX_SHOP_INSTANCES),
        "final_multisets": stages[str(MAX_SHOP_INSTANCES)][
            "unique_multisets"
        ],
        "stages": stages,
    }

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[SHOP-SPACE-DONE] total_prefix_multisets={total_states} "
        f"final_multisets={payload['final_multisets']} "
        f"ordered_sequences={payload['ordered_sequences_final']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
