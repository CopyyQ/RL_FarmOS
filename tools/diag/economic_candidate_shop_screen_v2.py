from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src"), str(ROOT)]

from kaggrl.v45_economics import ANIMAL_META, CROP_META, ECON_BASE_PRICE, strategy_snapshot
from kaggrl.v4_market_race import market_price
from tools.diag.economic_candidate_shop_coverage import make_reference_observation
from tools.diag.shop_space import enumerate_shop_multisets

CROPS = tuple(CROP_META)
ANIMALS = tuple(ANIMAL_META)
PRODUCT_TO_ANIMAL = {ANIMAL_META[a]["product"]: a for a in ANIMALS}
TARGET_CODE = {name: i for i, name in enumerate(("NONE",) + CROPS + ANIMALS)}


def _animal_payback(snapshot, observation, animal):
    ad = ANIMAL_META[animal]
    remaining_days = float(snapshot["remaining_days"])
    if remaining_days < float(ad["first"]) + 0.25:
        return {"roi": 0.0, "margin": -1.0, "revenue": 0.0, "cost": float(ad["cost"])}
    cycles = 1 + int(max(0.0, remaining_days - float(ad["first"])) // max(1.0, float(ad["interval"])))
    product = str(ad["product"])
    market = observation.get("market") or {}
    inv0 = float((market.get("inventory") or {}).get(product, 10000))
    future_sink = float(snapshot["future_sink"].get(product, 0.0))
    frac = min(1.0, float(ad["first"]) / max(1e-6, remaining_days))
    projected_inventory = max(0, int(round(inv0 - future_sink * frac)))
    revenue = 0.0
    inv = projected_inventory
    for _ in range(cycles):
        revenue += float(market_price(product, int(inv)))
        inv += 1
    wheat_price = float((market.get("prices") or {}).get("WHEAT", ECON_BASE_PRICE["WHEAT"]))
    feed_cost = remaining_days * max(3.0, 0.25 * wheat_price)
    total_cost = float(ad["cost"]) + feed_cost
    roi = revenue / max(1.0, total_cost)
    return {
        "roi": float(roi),
        "margin": float((revenue - total_cost) / max(1.0, total_cost)),
        "revenue": float(revenue),
        "cost": float(total_cost),
        "projected_inventory": int(projected_inventory),
        "cycles": int(cycles),
    }


def _viable_crops(snapshot, options):
    return [
        crop for crop in CROPS
        if bool(snapshot["crop_viable"].get(crop, False))
        and float(snapshot["demand"].get(crop, 0.0)) >= float(options.get("portfolio_min_shop_demand", 0.0))
        and float(snapshot["crop_undersupply_ratio"].get(crop, 0.0)) >= float(options.get("portfolio_min_undersupply_ratio", 0.0))
    ]


def _crop_target(snapshot, options, original):
    family = str(options.get("portfolio_objective", "best_crop_roi"))
    mode = str(options.get("portfolio_source_mode", "any"))
    viable = _viable_crops(snapshot, options)
    if original not in CROPS or not viable:
        return original
    if mode == "none":
        return original
    if mode == "wheat" and original != "WHEAT":
        return original
    if mode == "glutted" and float(snapshot["crop_projected_inventory"].get(original, 10000.0)) <= 10000.0:
        return original
    if mode == "nonbest":
        current_best = max(viable, key=lambda c: float(snapshot["crop_roi"].get(c, 0.0)))
        if original == current_best:
            return original
    pool = [c for c in viable if c != original]
    if not pool:
        return original
    if family == "scarcity_crop":
        key = lambda c: (float(snapshot["crop_undersupply_ratio"].get(c, 0.0)), float(snapshot["crop_roi"].get(c, 0.0)), c)
    elif family == "diversified_crop":
        penalty = float(options.get("portfolio_diversity_penalty", 0.2))
        key = lambda c: (float(snapshot["crop_roi"].get(c, 0.0)) / (1.0 + penalty * float(snapshot["demand"].get(c, 0.0))), float(snapshot["crop_undersupply_ratio"].get(c, 0.0)), c)
    elif family == "shop_specialist":
        key = lambda c: (float(snapshot["demand"].get(c, 0.0)), float(snapshot["crop_projected_price"].get(c, 0.0)), float(snapshot["crop_roi"].get(c, 0.0)), c)
    else:
        key = lambda c: (float(snapshot["crop_roi"].get(c, 0.0)), float(snapshot["crop_projected_price"].get(c, 0.0)), c)
    best = max(pool, key=key)
    old_roi = max(1e-6, float(snapshot["crop_roi"].get(original, 0.0)))
    new_roi = float(snapshot["crop_roi"].get(best, 0.0))
    return best if new_roi >= old_roi * float(options.get("portfolio_crop_improvement_ratio", 1.0)) else original


def _animal_target(snapshot, observation, options):
    forced = options.get("portfolio_forced_animal")
    min_roi = float(options.get("portfolio_animal_min_roi", 1.0))
    min_margin = float(options.get("portfolio_animal_payback_margin", 0.0))
    candidates = []
    for animal in ANIMALS:
        if forced and animal != forced:
            continue
        if not bool(snapshot["animal_viable"].get(animal, False)):
            continue
        payback = _animal_payback(snapshot, observation, animal)
        product = ANIMAL_META[animal]["product"]
        if payback["roi"] < min_roi or payback["margin"] < min_margin:
            continue
        candidates.append((payback["roi"], float(snapshot["demand"].get(product, 0.0)), animal))
    return max(candidates)[2] if candidates else "NONE"


def _decision(snapshot, observation, options, original):
    objective = str(options.get("portfolio_objective", "best_crop_roi"))
    crop = _crop_target(snapshot, options, original)
    if objective == "animal":
        return _animal_target(snapshot, observation, options)
    if objective != "hybrid":
        return crop
    animal = _animal_target(snapshot, observation, options)
    if animal == "NONE":
        return crop
    animal_roi = _animal_payback(snapshot, observation, animal)["roi"]
    crop_roi = float(snapshot["crop_roi"].get(crop, 0.0))
    return animal if animal_roi > crop_roi else crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(ROOT / "configs/tournament/economic_v2_raw.json"))
    ap.add_argument("--output", default=str(ROOT / "runs/economic_v2_shop_screen.json"))
    ap.add_argument("--selected", default=str(ROOT / "configs/tournament/economic_v2_selected.json"))
    ap.add_argument("--max-selected", type=int, default=100)
    ap.add_argument("--near-duplicate", type=float, default=0.997)
    args = ap.parse_args()

    payload = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    candidates = list(payload["candidates"])
    horizon_values = sorted({float((c.get("runtime_options") or {}).get("portfolio_horizon_extra_days", 0.0)) for c in candidates})
    states = []
    snapshots = {}
    for draws in range(1, 9):
        step = draws * 72
        for state in enumerate_shop_multisets(draws):
            obs = make_reference_observation(state["shops"], step)
            key = (draws, tuple(sorted(state["shops"].items())))
            states.append((key, state, obs))
            snapshots[key] = {h: strategy_snapshot(obs, projected_crop_value=True, projected_crop_horizon_extra_days=h) for h in horizon_values}

    full_states = [(key, state, obs) for key, state, obs in states if key[0] == 8]
    reports = []
    vectors = {}
    for idx, candidate in enumerate(candidates, 1):
        options = dict(candidate.get("runtime_options") or {})
        horizon = float(options.get("portfolio_horizon_extra_days", 0.0))
        targets = Counter()
        switch_mass = 0.0
        animal_mass = 0.0
        vec = bytearray()
        full_hasher = hashlib.sha256()
        for state_index, (key, state, obs) in enumerate(states):
            snap = snapshots[key][horizon]
            p = float(state["probability"])
            for original in CROPS:
                target = original if candidate.get("baseline") else _decision(snap, obs, options, original)
                targets[target] += p / len(CROPS)
                switch_mass += p / len(CROPS) * float(target != original)
                animal_mass += p / len(CROPS) * float(target in ANIMALS)
                code = TARGET_CODE[target]
                full_hasher.update(bytes((code,)))
                if state_index % 16 == 0:
                    vec.append(code)
        signature = full_hasher.hexdigest()
        vectors[candidate["id"]] = bytes(vec)
        reports.append({
            "id": candidate["id"], "family": candidate["family"], "baseline": bool(candidate.get("baseline")),
            "signature": signature, "switch_mass_sum_over_draws": switch_mass,
            "animal_mass_sum_over_draws": animal_mass, "targets": dict(sorted(targets.items())),
        })
        if idx % 25 == 0 or idx == len(candidates):
            print(f"[V2-SCREEN] {idx}/{len(candidates)} family={candidate['family']} targets={','.join(sorted(targets))}", flush=True)

    by_signature = {}
    exact_unique = []
    for candidate, report in zip(candidates, reports):
        if report["signature"] in by_signature:
            report["exact_duplicate_of"] = by_signature[report["signature"]]
            continue
        by_signature[report["signature"]] = candidate["id"]
        exact_unique.append(candidate)

    report_by_id = {r["id"]: r for r in reports}
    family_buckets = defaultdict(list)
    for c in exact_unique:
        if not c.get("baseline"):
            family_buckets[c["family"]].append(c)

    # Semantic hard gate: preserve at least one genuinely unique representative
    # from every surviving family. Rare regime specialists (especially WOOL)
    # must not be erased merely because they differ on a small probability mass.
    selected = [next(c for c in candidates if c.get("baseline"))]
    for family in sorted(family_buckets):
        bucket = family_buckets[family]
        if not bucket:
            continue
        representative = max(
            bucket,
            key=lambda c: (
                float(report_by_id[c["id"]]["animal_mass_sum_over_draws"]),
                float(report_by_id[c["id"]]["switch_mass_sum_over_draws"]),
                c["id"],
            ),
        )
        selected.append(representative)
        bucket.remove(representative)
    selected_vectors = [vectors[c["id"]] for c in selected]
    families = sorted(family_buckets)
    cursor = 0
    while len(selected) < args.max_selected and any(family_buckets.values()):
        family = families[cursor % len(families)]
        cursor += 1
        if not family_buckets[family]:
            continue
        chosen = None
        for candidate in list(family_buckets[family]):
            v = vectors[candidate["id"]]
            max_similarity = 0.0
            for sv in selected_vectors:
                same = sum(a == b for a, b in zip(v, sv)) / max(1, len(v))
                max_similarity = max(max_similarity, same)
                if max_similarity >= args.near_duplicate:
                    break
            if max_similarity < args.near_duplicate:
                chosen = candidate
                break
        if chosen is None:
            family_buckets[family].clear()
            continue
        family_buckets[family].remove(chosen)
        selected.append(chosen)
        selected_vectors.append(vectors[chosen["id"]])

    selected_ids = {c["id"] for c in selected}
    family_selected = Counter(c["family"] for c in selected)
    result = {
        "total_prefix_multisets": len(states), "full_8shop_multisets": len(full_states),
        "raw_candidate_count": len(candidates), "exact_unique_count": len(exact_unique),
        "selected_count": len(selected), "near_duplicate_similarity": args.near_duplicate,
        "selected_family_counts": dict(sorted(family_selected.items())), "reports": reports,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    selected_payload = dict(payload)
    selected_payload["schema"] = "farmos_candidate_manifest_v2_screened"
    selected_payload["screen"] = {k: result[k] for k in ("total_prefix_multisets", "raw_candidate_count", "exact_unique_count", "selected_count", "near_duplicate_similarity", "selected_family_counts")}
    selected_payload["candidates"] = [c for c in candidates if c["id"] in selected_ids]
    Path(args.selected).write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")
    print(f"[V2-SCREEN-DONE] states={len(states)} raw={len(candidates)} exact_unique={len(exact_unique)} selected={len(selected)} families={dict(family_selected)}", flush=True)


if __name__ == "__main__":
    main()
