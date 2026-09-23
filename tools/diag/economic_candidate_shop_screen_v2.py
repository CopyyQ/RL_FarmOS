from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src"), str(ROOT)]

from kaggrl.v45_economics import (
    ANIMAL_META,
    CROP_META,
    projected_animal_payback,
    strategy_snapshot,
)
from kaggrl.v4_market_race import market_price
from tools.diag.economic_candidate_shop_coverage import make_reference_observation
from tools.diag.shop_space import enumerate_shop_multisets

CROPS = tuple(CROP_META)
ANIMALS = tuple(ANIMAL_META)
PRODUCT_TO_ANIMAL = {ANIMAL_META[a]["product"]: a for a in ANIMALS}
TARGET_CODE = {name: i for i, name in enumerate(("NONE",) + CROPS + ANIMALS)}

# Supply stress is a separate deterministic axis from shop probability.
# Each regime preserves exact shop probability mass; regimes themselves are
# stress slices and are intentionally not assigned fake probabilities.
SUPPLY_REGIMES = (
    {"id": "neutral", "crop": None, "inventory_delta": 0},
    *tuple(
        {"id": f"realized_glut_{crop.lower()}", "crop": crop, "inventory_delta": 5000}
        for crop in CROPS
    ),
)


def _apply_supply_regime(observation, regime):
    if not regime.get("crop"):
        return observation
    obs = deepcopy(observation)
    crop = str(regime["crop"])
    inventory = obs["market"]["inventory"]
    prices = obs["market"]["prices"]
    inventory[crop] = max(
        0,
        int(inventory.get(crop, 10000)) + int(regime["inventory_delta"]),
    )
    prices[crop] = int(market_price(crop, inventory[crop]))
    return obs


def _screen_snapshot(observation, horizon, *, cache_animal_payback):
    snap = strategy_snapshot(
        observation,
        projected_crop_value=True,
        projected_crop_horizon_extra_days=horizon,
    )
    if cache_animal_payback:
        snap["_screen_animal_payback"] = {
            animal: projected_animal_payback(snap, observation, animal)
            for animal in ANIMALS
        }
    return snap


def _animal_payback(snapshot, observation, animal):
    cached = snapshot.get("_screen_animal_payback", {}).get(str(animal))
    if cached is not None:
        return cached
    return projected_animal_payback(snapshot, observation, animal)


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


def _animal_target(snapshot, observation, options, original, paybacks=None):
    original = str(original)
    if paybacks is None:
        paybacks = {}
    forced = options.get("portfolio_forced_animal")
    min_roi = float(options.get("portfolio_animal_min_roi", 1.0))
    min_margin = float(options.get("portfolio_animal_payback_margin", 0.0))
    candidates = []
    for animal in ANIMALS:
        if forced and animal != forced:
            continue
        if not bool(snapshot["animal_viable"].get(animal, False)):
            continue
        if animal not in paybacks:
            paybacks[animal] = _animal_payback(snapshot, observation, animal)
        payback = paybacks[animal]
        product = ANIMAL_META[animal]["product"]
        if payback["roi"] < min_roi or payback["margin"] < min_margin:
            continue
        candidates.append((
            payback["roi"],
            float(snapshot["demand"].get(product, 0.0)),
            animal,
        ))
    if not candidates:
        return original
    best = max(candidates)[2]
    if best == original:
        return original
    if not forced:
        if original not in paybacks:
            paybacks[original] = _animal_payback(snapshot, observation, original)
        if best not in paybacks:
            paybacks[best] = _animal_payback(snapshot, observation, best)
        old = paybacks[original]
        new = paybacks[best]
        if float(new["roi"]) < float(old["roi"]):
            return original
    return best


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
            base_obs = make_reference_observation(state["shops"], step)
            shop_key = (draws, tuple(sorted(state["shops"].items())))
            for regime in SUPPLY_REGIMES:
                obs = _apply_supply_regime(base_obs, regime)
                key = (shop_key, str(regime["id"]))
                states.append((key, state, obs, regime))
                cache_animal = str(regime["id"]) == "neutral"
                snapshots[key] = {
                    h: _screen_snapshot(
                        obs,
                        h,
                        cache_animal_payback=cache_animal,
                    )
                    for h in horizon_values
                }

    neutral_states = [
        row for row in states if str(row[3]["id"]) == "neutral"
    ]
    full_states = [row for row in states if row[0][0][0] == 8]
    reports = []
    vectors = {}
    for idx, candidate in enumerate(candidates, 1):
        options = dict(candidate.get("runtime_options") or {})
        horizon = float(options.get("portfolio_horizon_extra_days", 0.0))
        crop_targets = Counter()
        animal_targets = Counter()
        crop_switch_targets = Counter()
        animal_switch_targets = Counter()
        crop_switch_mass = 0.0
        animal_switch_mass = 0.0
        per_regime_crop_switch = defaultdict(float)
        per_regime_animal_switch = defaultdict(float)
        objective = str(options.get("portfolio_objective", "best_crop_roi"))
        crop_enabled = objective != "animal"
        animal_enabled = objective in ("animal", "hybrid")
        family = str(candidate.get("family", ""))
        stress_scope = family == "B_glut_to_scarcity"
        candidate_states = states if stress_scope else neutral_states
        screen_scope = "neutral_plus_glut" if stress_scope else "neutral"
        vec = bytearray()
        full_hasher = hashlib.sha256()
        for state_index, (key, state, obs, regime) in enumerate(candidate_states):
            snap = snapshots[key][horizon]
            p = float(state["probability"])
            regime_id = str(regime["id"])
            for original in CROPS:
                target = original
                if not candidate.get("baseline") and crop_enabled:
                    target = _crop_target(snap, options, original)
                crop_targets[target] += p / len(CROPS)
                switched = p / len(CROPS) * float(target != original)
                crop_switch_mass += switched
                per_regime_crop_switch[regime_id] += switched
                if target != original:
                    crop_switch_targets[target] += p / len(CROPS)
                code = TARGET_CODE[target]
                full_hasher.update(bytes((code,)))
                if state_index % 128 == 0:
                    vec.append(code)
            animal_paybacks = {}
            for original in ANIMALS:
                target = original
                if not candidate.get("baseline") and animal_enabled:
                    target = _animal_target(
                        snap,
                        obs,
                        options,
                        original,
                        animal_paybacks,
                    )
                animal_targets[target] += p / len(ANIMALS)
                switched = p / len(ANIMALS) * float(target != original)
                animal_switch_mass += switched
                per_regime_animal_switch[regime_id] += switched
                if target != original:
                    animal_switch_targets[target] += p / len(ANIMALS)
                code = TARGET_CODE[target]
                full_hasher.update(bytes((code,)))
                if state_index % 128 == 0:
                    vec.append(code)
        signature = full_hasher.hexdigest()
        vectors[candidate["id"]] = bytes(vec)
        all_targets = set(crop_targets) | set(animal_targets)
        reports.append({
            "id": candidate["id"], "family": candidate["family"], "baseline": bool(candidate.get("baseline")),
            "signature": signature,
            "screen_scope": screen_scope,
            "screen_state_count": len(candidate_states),
            "crop_switch_mass_sum_over_draws": crop_switch_mass,
            "animal_switch_mass_sum_over_draws": animal_switch_mass,
            "switch_mass_sum_over_draws": crop_switch_mass + animal_switch_mass,
            "crop_targets": dict(sorted(crop_targets.items())),
            "animal_targets": dict(sorted(animal_targets.items())),
            "crop_switch_targets": dict(sorted(crop_switch_targets.items())),
            "animal_switch_targets": dict(sorted(animal_switch_targets.items())),
            "per_supply_regime_crop_switch_mass": dict(sorted(per_regime_crop_switch.items())),
            "per_supply_regime_animal_switch_mass": dict(sorted(per_regime_animal_switch.items())),
            "targets": sorted(all_targets),
        })
        if idx % 25 == 0 or idx == len(candidates):
            print(f"[V2-SCREEN] {idx}/{len(candidates)} family={candidate['family']} targets={','.join(sorted(all_targets))}", flush=True)

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
                float(report_by_id[c["id"]]["animal_switch_mass_sum_over_draws"]),
                float(report_by_id[c["id"]]["switch_mass_sum_over_draws"]),
                c["id"],
            ),
        )
        selected.append(representative)
        bucket.remove(representative)
    selected_vectors = [
        (report_by_id[c["id"]]["screen_scope"], vectors[c["id"]])
        for c in selected
    ]
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
            candidate_scope = report_by_id[candidate["id"]]["screen_scope"]
            max_similarity = 0.0
            for selected_scope, sv in selected_vectors:
                if selected_scope != candidate_scope:
                    continue
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
        selected_vectors.append((
            report_by_id[chosen["id"]]["screen_scope"],
            vectors[chosen["id"]],
        ))

    selected_ids = {c["id"] for c in selected}
    family_selected = Counter(c["family"] for c in selected)
    result = {
        "shop_prefix_multisets": len(states) // len(SUPPLY_REGIMES),
        "supply_regime_count": len(SUPPLY_REGIMES),
        "supply_regimes": list(SUPPLY_REGIMES),
        "total_state_regimes": len(states),
        "full_8shop_state_regimes": len(full_states),
        "raw_candidate_count": len(candidates), "exact_unique_count": len(exact_unique),
        "selected_count": len(selected), "near_duplicate_similarity": args.near_duplicate,
        "selected_family_counts": dict(sorted(family_selected.items())), "reports": reports,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    selected_payload = dict(payload)
    selected_payload["schema"] = "farmos_candidate_manifest_v2_screened"
    selected_payload["screen"] = {k: result[k] for k in ("shop_prefix_multisets", "supply_regime_count", "total_state_regimes", "raw_candidate_count", "exact_unique_count", "selected_count", "near_duplicate_similarity", "selected_family_counts")}
    selected_payload["candidates"] = [c for c in candidates if c["id"] in selected_ids]
    Path(args.selected).write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")
    print(
        f"[V2-SCREEN-DONE] shop_states={len(states) // len(SUPPLY_REGIMES)} "
        f"supply_regimes={len(SUPPLY_REGIMES)} state_regimes={len(states)} "
        f"raw={len(candidates)} exact_unique={len(exact_unique)} "
        f"selected={len(selected)} families={dict(family_selected)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
