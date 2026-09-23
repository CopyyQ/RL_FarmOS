from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src"), str(ROOT)]

from kaggrl.v45_economics import (
    ECON_BASE_PRICE,
    ECON_SHOP_PRODUCTS,
    ECON_PRODUCTS,
    MARKET_CAPACITY,
)
from tools.diag.stage_money_forensics import run_game


KEY_PRODUCTS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
)


def _shop_sink_per_day(shops):
    sink = {product: 0.0 for product in ECON_PRODUCTS}
    for shop in list(shops or []):
        products = ECON_SHOP_PRODUCTS.get(str(shop), ())
        multiplier = 2.0 if len(products) == 1 else 1.0
        for product in products:
            sink[product] += 6.0 * multiplier
    for product in KEY_PRODUCTS:
        sink[product] += 1.0
    return sink


def _sum_product_metric(game, side, bucket, field):
    totals = defaultdict(float)
    for phase in game["phases"]:
        rows = phase["market_cash"][side].get(bucket, {})
        for key, value in rows.items():
            if bucket == "sales":
                product = str(key)
            else:
                kind, _, product = str(key).partition(":")
                product = product or kind
            totals[product] += float(value.get(field, 0) or 0)
    return dict(totals)


def _safe_ratio(numer, denom):
    return float(numer) / max(1e-9, float(denom))


def summarize_game(game):
    final = game["phases"][-1]["end"]
    our_final = final["ours"]
    shops = list(our_final.get("shops") or [])
    shop_counts = Counter(shops)
    sink = _shop_sink_per_day(shops)

    phase_prices = {}
    max_prices = {product: 0.0 for product in KEY_PRODUCTS}
    for phase in game["phases"]:
        prices = dict(phase["end"]["ours"].get("prices") or {})
        phase_prices[phase["phase"]] = {
            product: int(prices.get(product, 0) or 0)
            for product in KEY_PRODUCTS
        }
        for product in KEY_PRODUCTS:
            max_prices[product] = max(
                max_prices[product],
                float(prices.get(product, 0) or 0),
            )

    final_prices = phase_prices[game["phases"][-1]["phase"]]
    price_ratio = {
        product: _safe_ratio(final_prices[product], ECON_BASE_PRICE[product])
        for product in KEY_PRODUCTS
    }
    max_price_ratio = {
        product: _safe_ratio(max_prices[product], ECON_BASE_PRICE[product])
        for product in KEY_PRODUCTS
    }
    scarcity_product = max(
        KEY_PRODUCTS,
        key=lambda p: (max_price_ratio[p], sink[p], p),
    )
    sink_pressure = {
        product: _safe_ratio(sink[product], MARKET_CAPACITY[product] / 24.0)
        for product in KEY_PRODUCTS
    }
    sink_product = max(
        KEY_PRODUCTS,
        key=lambda p: (sink_pressure[p], sink[p], p),
    )

    our_revenue = _sum_product_metric(game, "ours", "sales", "revenue")
    opp_revenue = _sum_product_metric(game, "opp", "sales", "revenue")
    revenue_gap = {
        product: float(our_revenue.get(product, 0.0))
        - float(opp_revenue.get(product, 0.0))
        for product in KEY_PRODUCTS
    }
    biggest_revenue_loss_product = min(
        KEY_PRODUCTS,
        key=lambda p: (revenue_gap[p], p),
    )

    phase_cash_changes = {
        phase["phase"]: int(phase["cash"]["gap_change"])
        for phase in game["phases"]
    }
    phase_econ_end = {
        phase["phase"]: int(phase["economic"]["horizon_gap_end"])
        for phase in game["phases"]
    }

    portfolio = {}
    for phase in game["phases"]:
        portfolio[phase["phase"]] = {
            "ours_crops": dict(phase["end"]["ours"]["farm"].get("crops") or {}),
            "opp_crops": dict(phase["end"]["opp"]["farm"].get("crops") or {}),
            "ours_animals": dict(
                phase["end"]["ours"]["farm"].get("animals") or {}
            ),
            "opp_animals": dict(
                phase["end"]["opp"]["farm"].get("animals") or {}
            ),
        }

    return {
        "seed": int(game["seed"]),
        "seat": int(game["our_seat"]),
        "margin": int(game["final_margin"]),
        "shops": dict(sorted(shop_counts.items())),
        "sink_per_day": {
            product: float(sink[product]) for product in KEY_PRODUCTS
        },
        "sink_pressure": sink_pressure,
        "sink_product": sink_product,
        "final_prices": final_prices,
        "max_prices": max_prices,
        "final_price_ratio": price_ratio,
        "max_price_ratio": max_price_ratio,
        "scarcity_product": scarcity_product,
        "our_revenue": {
            product: float(our_revenue.get(product, 0.0))
            for product in KEY_PRODUCTS
        },
        "opp_revenue": {
            product: float(opp_revenue.get(product, 0.0))
            for product in KEY_PRODUCTS
        },
        "revenue_gap": revenue_gap,
        "biggest_revenue_loss_product": biggest_revenue_loss_product,
        "phase_cash_gap_change": phase_cash_changes,
        "phase_economic_gap_end": phase_econ_end,
        "portfolio": portfolio,
    }


def _worker(args):
    seed, seat = args
    return summarize_game(run_game(int(seed), int(seat)))


def _mean(values):
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def _percentile(values, q):
    xs = sorted(float(v) for v in values)
    if not xs:
        return 0.0
    pos = (len(xs) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def aggregate(rows):
    margins = [int(row["margin"]) for row in rows]
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["scarcity_product"])].append(row)

    regimes = {}
    for product, members in sorted(grouped.items()):
        ms = [int(row["margin"]) for row in members]
        rev_gap = [
            float(row["revenue_gap"].get(product, 0.0))
            for row in members
        ]
        regimes[product] = {
            "games": len(members),
            "mean_margin": _mean(ms),
            "median_margin": statistics.median(ms) if ms else 0.0,
            "worst_margin": min(ms) if ms else 0,
            "mean_product_revenue_gap": _mean(rev_gap),
            "mean_max_price_ratio": _mean(
                row["max_price_ratio"][product] for row in members
            ),
            "examples": [
                {
                    "seed": row["seed"],
                    "seat": row["seat"],
                    "margin": row["margin"],
                    "shops": row["shops"],
                    "max_price_ratio": row["max_price_ratio"][product],
                    "revenue_gap": row["revenue_gap"][product],
                }
                for row in sorted(members, key=lambda r: r["margin"])[:5]
            ],
        }

    return {
        "games": len(rows),
        "mean_margin": _mean(margins),
        "median_margin": statistics.median(margins) if margins else 0.0,
        "p10_margin": _percentile(margins, 0.10),
        "worst_margin": min(margins) if margins else 0,
        "below_20k": sum(x <= -20000 for x in margins),
        "below_30k": sum(x <= -30000 for x in margins),
        "below_50k": sum(x <= -50000 for x in margins),
        "scarcity_regimes": regimes,
        "largest_losses": [
            {
                "seed": row["seed"],
                "seat": row["seat"],
                "margin": row["margin"],
                "scarcity_product": row["scarcity_product"],
                "sink_product": row["sink_product"],
                "biggest_revenue_loss_product": row[
                    "biggest_revenue_loss_product"
                ],
                "shops": row["shops"],
            }
            for row in sorted(rows, key=lambda r: r["margin"])[:10]
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=23242000)
    ap.add_argument("--seed-count", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--both-seats", action="store_true", default=True)
    ap.add_argument(
        "--output",
        default=str(ROOT / "runs" / "market_regime_atlas.json"),
    )
    args = ap.parse_args()

    games = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        if args.both_seats:
            games.extend(((seed, 0), (seed, 1)))
        else:
            games.append((seed, (seed + 1205) & 1))

    rows = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(_worker, game): game for game in games}
        done = 0
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            done += 1
            print(
                f"[ATLAS] {done:03d}/{len(games):03d} "
                f"seed={row['seed']} seat={row['seat']} "
                f"margin={row['margin']:+d} "
                f"scarcity={row['scarcity_product']} "
                f"loss_product={row['biggest_revenue_loss_product']}",
                flush=True,
            )

    rows.sort(key=lambda r: (r["seed"], r["seat"]))
    payload = {
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seed_count),
        "both_seats": bool(args.both_seats),
        "summary": aggregate(rows),
        "games": rows,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "[ATLAS-DONE]",
        json.dumps(payload["summary"], ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
