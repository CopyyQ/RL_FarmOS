from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src"), str(ROOT)]

from kaggrl.v45_economics import (
    CROP_META,
    ECON_BASE_PRICE,
    ECON_PRODUCTS,
    ECON_SHOP_PRODUCTS,
    TOWN_CENTER_PRODUCTS,
    strategy_snapshot,
)
from kaggrl.v4_market_race import MARKET_PARAMS, market_price
from tools.diag.shop_space import enumerate_shop_multisets


def make_reference_observation(shops, step):
    draws = max(1, sum(int(x) for x in shops.values()))
    # Conditional on an unordered multiset, every assignment of its shop
    # instances to reveal positions is equiprobable. A shop opened at reveal j
    # has about 18 * (draws-j) four-step consumption events before the d-th
    # reveal. The conditional expected exposure per instance is therefore
    # 9*(draws-1) events. This preserves exact multiset probability coverage
    # while incorporating the market history that a neutral I0 snapshot misses.
    expected_shop_events = 9.0 * max(0, draws - 1)
    center_events = max(0, int(step) // 24)
    sink = {product: 0.0 for product in ECON_PRODUCTS}
    for shop, count in shops.items():
        products = ECON_SHOP_PRODUCTS.get(str(shop), ())
        mult = 2.0 if len(products) == 1 else 1.0
        for product in products:
            sink[product] += (
                float(count) * expected_shop_events * mult
            )
    for product in TOWN_CENTER_PRODUCTS:
        sink[product] += float(center_events)

    inventory = {}
    prices = {}
    for product in ECON_PRODUCTS:
        i0 = int(MARKET_PARAMS[product]["I0"])
        inventory[product] = max(
            0, int(round(i0 - sink.get(product, 0.0)))
        )
        prices[product] = int(
            market_price(product, inventory[product])
        )
    farm = {
        "money": 10000,
        "hands": [],
        "tiles": [[None] * 4 for _ in range(4)],
    }
    rival = {
        "money": 10000,
        "hands": [],
        "tiles": [[None] * 4 for _ in range(4)],
    }
    expanded = []
    for shop, count in sorted(shops.items()):
        expanded.extend([shop] * int(count))
    return {
        "step": int(step),
        "day": int(step) // 24,
        "player": 0,
        "town": {"unlocked_shops": expanded},
        "market": {"inventory": inventory, "prices": prices},
        "farms": [farm, rival],
        "private": {
            "shed": {},
            "seeds": {
                "WHEAT": 5,
                "CARROT": 5,
                "TOMATO": 5,
                "STRAWBERRY": 5,
                "MELON": 5,
            },
            "inventories": [{}],
        },
    }


def choose_from_snapshot(snapshot, options, original="WHEAT"):
    if not bool(options.get("enable_portfolio_switch", False)):
        return original
    if int(snapshot["step"]) < int(
        options.get("portfolio_activation_step", 144)
    ):
        return original

    mode = str(options.get("portfolio_source_mode", "any"))
    viable = [
        crop
        for crop in CROP_META
        if bool(snapshot["crop_viable"].get(crop, False))
        and float(snapshot["demand"].get(crop, 0.0))
        >= float(options.get("portfolio_min_shop_demand", 1.0))
        and float(snapshot["crop_undersupply_ratio"].get(crop, 0.0))
        >= float(options.get("portfolio_min_undersupply_ratio", 0.0))
    ]
    if mode == "wheat" and original != "WHEAT":
        return original
    if mode == "nonbest" and viable:
        current_best = max(
            viable,
            key=lambda c: float(snapshot["crop_roi"].get(c, 0.0)),
        )
        if original == current_best:
            return original

    candidates = [crop for crop in viable if crop != original]
    if not candidates:
        return original
    best = max(
        candidates,
        key=lambda c: (
            float(snapshot["crop_roi"].get(c, 0.0)),
            float(snapshot["crop_projected_price"].get(c, 0.0)),
            -float(CROP_META[c]["first"]),
            c,
        ),
    )
    old_roi = max(
        1e-6, float(snapshot["crop_roi"].get(original, 0.0))
    )
    new_roi = float(snapshot["crop_roi"].get(best, 0.0))
    ratio = float(options.get("portfolio_crop_improvement_ratio", 1.30))
    return best if new_roi >= old_roi * ratio else original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidates",
        default=str(
            ROOT / "configs" / "tournament" / "economic100.json"
        ),
    )
    ap.add_argument(
        "--output",
        default=str(
            ROOT / "runs" / "economic_shop_coverage.json"
        ),
    )
    args = ap.parse_args()

    payload = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    candidates = list(payload["candidates"])

    horizon_values = sorted(
        {
            float(
                candidate.get("runtime_options", {}).get(
                    "portfolio_horizon_extra_days", 0.0
                )
            )
            for candidate in candidates
        }
    )
    snapshots = {}
    states = []
    total_states = 0
    for draws in range(1, 9):
        step = draws * 72
        for state in enumerate_shop_multisets(draws):
            key = (
                draws,
                tuple(sorted(state["shops"].items())),
            )
            obs = make_reference_observation(state["shops"], step)
            snapshots[key] = {
                horizon: strategy_snapshot(
                    obs,
                    projected_crop_value=True,
                    projected_crop_horizon_extra_days=horizon,
                )
                for horizon in horizon_values
            }
            states.append((key, state))
            total_states += 1

    reports = []
    for index, candidate in enumerate(candidates, 1):
        options = dict(candidate.get("runtime_options") or {})
        horizon = float(
            options.get("portfolio_horizon_extra_days", 0.0)
        )
        per_draw = {}
        all_targets = set()
        for draws in range(1, 9):
            target_mass = defaultdict(float)
            switched_mass = 0.0
            state_count = 0
            for key, state in states:
                if key[0] != draws:
                    continue
                snap = snapshots[key][horizon]
                target = choose_from_snapshot(
                    snap, options, original="WHEAT"
                )
                probability = float(state["probability"])
                target_mass[target] += probability
                switched_mass += probability * float(target != "WHEAT")
                all_targets.add(target)
                state_count += 1
            per_draw[str(draws)] = {
                "states": state_count,
                "switch_probability_mass": switched_mass,
                "target_probability_mass": dict(
                    sorted(target_mass.items())
                ),
            }

        report = {
            "id": candidate["id"],
            "baseline": bool(candidate.get("baseline")),
            "unique_targets": sorted(all_targets),
            "per_draw": per_draw,
        }
        reports.append(report)
        print(
            f"[SHOP-COVERAGE] {index:03d}/{len(candidates):03d} "
            f"{candidate['id']} targets={','.join(sorted(all_targets))}",
            flush=True,
        )

    result = {
        "total_prefix_multisets": total_states,
        "candidate_count": len(candidates),
        "evaluations": total_states * len(candidates),
        "horizon_values": horizon_values,
        "candidates": reports,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[SHOP-COVERAGE-DONE] states={total_states} "
        f"candidates={len(candidates)} "
        f"evaluations={result['evaluations']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
