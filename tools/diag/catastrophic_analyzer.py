from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


PRIMARY_PRIORITY = (
    "LATE_VALUE_REVEAL",
    "OPPONENT_ENDGAME_SNOWBALL",
    "ROUTE_STAGNATION",
    "TRUE_LIQUIDITY_COLLAPSE",
    "ENDGAME_SPEND",
    "STORAGE_PRESSURE",
    "PRODUCTION_LOSS",
    "WATER_MISS",
    "MARKET_HOLD_LOW_CASH",
    "STRANDED_INVENTORY",
    "ANIMAL_ESCAPE",
    "UNCLASSIFIED",
)


def load_cases(inputs):
    rows = []
    for source in inputs:
        path = Path(source)
        files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for file_path in files:
            if not file_path.exists():
                continue
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    row["_source"] = str(file_path)
                    rows.append(row)
    return rows


def _first_at_or_below(trace, key, threshold):
    for row in trace:
        try:
            if float(row.get(key, 0)) <= float(threshold):
                return int(row.get("step", -1))
        except Exception:
            pass
    return None


def _money_at_or_before(trace, step, key):
    candidates = [row for row in trace if int(row.get("step", -1)) <= int(step)]
    if not candidates:
        return 0
    return int(candidates[-1].get(key, 0) or 0)


def _longest_route_streak(trace):
    longest = current = 0
    previous = object()
    changes = 0
    for row in trace:
        route = row.get("route_action", -1)
        if route == previous:
            current += 1
        else:
            if longest:
                changes += 1
            previous = route
            current = 1
        longest = max(longest, current)
    return longest, max(0, changes - 1)


def _true_liquidity_collapse(trace, peak_cash):
    """Ignore normal opening investment dips.

    A liquidity collapse must happen after shop reveal (step 144), stay near
    empty for at least one full day, and fail to recover promptly.
    """
    floor = max(1000, int(0.05 * max(1, peak_cash)))
    late = [row for row in trace if int(row.get("step", -1)) >= 144]
    low_steps = [
        int(row.get("step", -1))
        for row in late
        if int(row.get("own_money", 0)) <= floor
    ]
    if not low_steps:
        return False, {}
    first = low_steps[0]
    sustained = sum(
        first <= int(row.get("step", -1)) < first + 24
        and int(row.get("own_money", 0)) <= floor
        for row in late
    )
    recovered = any(
        first < int(row.get("step", -1)) <= first + 48
        and int(row.get("own_money", 0)) >= 0.20 * peak_cash
        for row in late
    )
    return sustained >= 12 and not recovered, {
        "first_low_step": first,
        "low_steps_next_day": sustained,
        "floor": floor,
        "recovered_48": recovered,
    }


def summarize_case(case):
    trace = list(case.get("trace") or [])
    if not trace:
        return {
            "seed": int(case.get("seed", 0)),
            "seat": int(case.get("seat", 0)),
            "margin": int(case.get("margin", 0)),
            "primary_signal": "UNCLASSIFIED",
            "signals": ["UNCLASSIFIED"],
            "reason": "missing trace",
        }

    own = [int(row.get("own_money", 0)) for row in trace]
    rival = [int(row.get("rival_money", 0)) for row in trace]
    margin_path = [int(row.get("margin", 0)) for row in trace]
    peak_cash = max(own)
    min_cash = min(own)
    final_margin = int(case.get("margin", margin_path[-1]))

    step_day24 = min(575, int(trace[-1].get("step", 0)))
    own_day24 = _money_at_or_before(trace, step_day24, "own_money")
    rival_day24 = _money_at_or_before(trace, step_day24, "rival_money")
    margin_day24 = own_day24 - rival_day24
    own_late_growth = own[-1] - own_day24
    rival_late_growth = rival[-1] - rival_day24
    late_growth_advantage = rival_late_growth - own_late_growth

    last_96 = [row for row in trace if int(row.get("step", 0)) >= int(trace[-1]["step"]) - 95]
    if last_96:
        own_96 = own[-1] - int(last_96[0].get("own_money", own[-1]))
        rival_96 = rival[-1] - int(last_96[0].get("rival_money", rival[-1]))
    else:
        own_96 = rival_96 = 0

    route_streak, route_changes = _longest_route_streak(trace)
    market_counts = Counter()
    unit_counts = Counter()
    market_modes = Counter()
    endgame_capex = []
    for row in trace:
        step = int(row.get("step", -1))
        market_modes[str(row.get("market_mode", "KEEP_ROUTE"))] += 1
        for op, count in dict(row.get("market_counts") or {}).items():
            market_counts[str(op)] += int(count)
            if step >= 600 and str(op) in {
                "HIRE", "BUY_LAND", "BUY_ANIMAL", "BUY_SEED", "BUY_PRODUCT"
            }:
                endgame_capex.extend([step] * int(count))
        for op, count in dict(row.get("unit_counts") or {}).items():
            unit_counts[str(op)] += int(count)

    storage_peak = max(int(row.get("shed_total", 0)) for row in trace)
    final_storage = int(trace[-1].get("shed_total", 0))
    plant_deaths = sum(int(row.get("plant_deaths", 0)) for row in trace)
    animal_escapes = sum(int(row.get("animal_escapes", 0)) for row in trace)
    critical_water = sum(
        int(row.get("critical_water_opportunities", 0)) for row in trace
    )
    water_success = sum(
        int(row.get("water_success_actions", 0)) for row in trace
    )

    first_10k = _first_at_or_below(trace, "margin", -10000)
    first_20k = _first_at_or_below(trace, "margin", -20000)
    first_30k = _first_at_or_below(trace, "margin", -30000)

    signals = []
    evidence = {}

    # The strongest signature from seed 23238530: cash looks competitive around
    # day 24, then hidden productive/inventory value is converted by the rival.
    if margin_day24 >= -5000 and final_margin <= -30000:
        signals.append("LATE_VALUE_REVEAL")
        evidence["LATE_VALUE_REVEAL"] = {
            "margin_day24": margin_day24,
            "final_margin": final_margin,
            "own_late_growth": own_late_growth,
            "rival_late_growth": rival_late_growth,
            "late_growth_advantage": late_growth_advantage,
        }

    if (rival_96 - own_96) >= 15000 or late_growth_advantage >= 20000:
        signals.append("OPPONENT_ENDGAME_SNOWBALL")
        evidence["OPPONENT_ENDGAME_SNOWBALL"] = {
            "own_last96_growth": own_96,
            "rival_last96_growth": rival_96,
            "late_growth_advantage": late_growth_advantage,
        }

    # A route that persists for >=6 days while the game later catastrophically
    # diverges is worth auditing. This is a diagnostic signal, not proof that
    # the route itself is invalid.
    if route_streak >= 144 and final_margin <= -30000:
        signals.append("ROUTE_STAGNATION")
        evidence["ROUTE_STAGNATION"] = {
            "longest_route_streak": route_streak,
            "route_changes": route_changes,
        }

    liquidity, liquidity_evidence = _true_liquidity_collapse(trace, peak_cash)
    if liquidity:
        signals.append("TRUE_LIQUIDITY_COLLAPSE")
        evidence["TRUE_LIQUIDITY_COLLAPSE"] = liquidity_evidence

    if endgame_capex:
        signals.append("ENDGAME_SPEND")
        evidence["ENDGAME_SPEND"] = {
            "orders": len(endgame_capex),
            "first_step": min(endgame_capex),
        }

    if storage_peak >= 90:
        signals.append("STORAGE_PRESSURE")
        evidence["STORAGE_PRESSURE"] = {"peak_shed": storage_peak}

    if plant_deaths >= 25:
        signals.append("PRODUCTION_LOSS")
        evidence["PRODUCTION_LOSS"] = {"plant_deaths": plant_deaths}

    if critical_water >= 8 and water_success < 0.70 * critical_water:
        signals.append("WATER_MISS")
        evidence["WATER_MISS"] = {
            "critical": critical_water,
            "success": water_success,
        }

    low_cash_hold = sum(
        int(row.get("step", 0)) >= 144
        and str(row.get("market_mode", "")) == "HOLD_SALES"
        and int(row.get("own_money", 0)) <= max(2000, int(0.05 * peak_cash))
        for row in trace
    )
    if low_cash_hold >= 12:
        signals.append("MARKET_HOLD_LOW_CASH")
        evidence["MARKET_HOLD_LOW_CASH"] = {"steps": low_cash_hold}

    if final_storage >= 50 and final_margin < 0:
        signals.append("STRANDED_INVENTORY")
        evidence["STRANDED_INVENTORY"] = {"final_shed": final_storage}

    if animal_escapes > 0:
        signals.append("ANIMAL_ESCAPE")
        evidence["ANIMAL_ESCAPE"] = {"animal_escapes": animal_escapes}

    if not signals:
        signals = ["UNCLASSIFIED"]

    primary = next(
        (name for name in PRIMARY_PRIORITY if name in signals),
        "UNCLASSIFIED",
    )
    return {
        "seed": int(case.get("seed", 0)),
        "seat": int(case.get("seat", 0)),
        "iteration": int(case.get("iteration", 0)),
        "phase": str(case.get("phase", "")),
        "margin": final_margin,
        "primary_signal": primary,
        "signals": signals,
        "evidence": evidence,
        "first_below_10k": first_10k,
        "first_below_20k": first_20k,
        "first_below_30k": first_30k,
        "margin_day24": margin_day24,
        "peak_cash": peak_cash,
        "min_cash": min_cash,
        "own_late_growth": own_late_growth,
        "rival_late_growth": rival_late_growth,
        "late_growth_advantage": late_growth_advantage,
        "storage_peak": storage_peak,
        "final_storage": final_storage,
        "plant_deaths": plant_deaths,
        "animal_escapes": animal_escapes,
        "route_changes": route_changes,
        "longest_route_streak": route_streak,
        "market_orders": dict(market_counts),
        "unit_ops": dict(unit_counts),
        "market_mode_steps": dict(market_modes),
        "source": case.get("_source", ""),
    }


def aggregate(rows):
    if not rows:
        return {"cases": 0, "primary": {}, "signals": {}}
    primary = Counter(row["primary_signal"] for row in rows)
    signals = Counter(
        signal for row in rows for signal in row.get("signals", [])
    )
    first20 = [
        row["first_below_20k"] for row in rows
        if row["first_below_20k"] is not None
    ]
    return {
        "cases": len(rows),
        "mean_margin": statistics.fmean(row["margin"] for row in rows),
        "worst_margin": min(row["margin"] for row in rows),
        "primary": dict(primary.most_common()),
        "signals": dict(signals.most_common()),
        "mean_first_below_20k": (
            statistics.fmean(first20) if first20 else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Catastrophe JSONL files or directories from winner_train.py",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    cases = load_cases(args.inputs)
    rows = [summarize_case(case) for case in cases]
    rows.sort(key=lambda row: row["margin"])
    report = {
        "summary": aggregate(rows),
        "worst_cases": rows[: max(1, int(args.top))],
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
