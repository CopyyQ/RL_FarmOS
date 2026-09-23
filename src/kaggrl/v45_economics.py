from __future__ import annotations

import math
from typing import Any

from .v4_market_race import market_price

ECON_SHOPS = (
    "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
    "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET",
)
ECON_PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
ECON_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
ECON_BASE_PRICE = {
    "WHEAT": 25.0, "CARROT": 35.0, "TOMATO": 60.0,
    "STRAWBERRY": 120.0, "MELON": 250.0, "EGG": 50.0,
    "MILK": 160.0, "WOOL": 200.0, "FERTILIZER": 100.0,
}
CROP_META = {
    "WHEAT": {"seed": 10.0, "first": 2.0, "max_day": 4.0, "interval": 0.0, "max_yield": 6.0, "ongoing": False},
    "CARROT": {"seed": 20.0, "first": 2.0, "max_day": 3.0, "interval": 0.0, "max_yield": 4.0, "ongoing": False},
    "TOMATO": {"seed": 50.0, "first": 8.0, "max_day": 8.0, "interval": 1.0, "max_yield": 4.0, "ongoing": True},
    "STRAWBERRY": {"seed": 100.0, "first": 10.0, "max_day": 10.0, "interval": 2.0, "max_yield": 4.0, "ongoing": True},
    "MELON": {"seed": 80.0, "first": 10.0, "max_day": 12.0, "interval": 0.0, "max_yield": 6.0, "ongoing": False},
}
ANIMAL_META = {
    "GOOSE": {"cost": 300.0, "first": 4.0, "interval": 1.0, "product": "EGG", "structure": "COOP"},
    "COW": {"cost": 400.0, "first": 8.0, "interval": 2.0, "product": "MILK", "structure": "PASTURE"},
    "SHEEP": {"cost": 500.0, "first": 6.0, "interval": 3.0, "product": "WOOL", "structure": "PASTURE"},
}
MARKET_CAPACITY = {
    "WHEAT": 400.0, "CARROT": 450.0, "TOMATO": 200.0,
    "STRAWBERRY": 100.0, "MELON": 300.0, "EGG": 332.0,
    "MILK": 122.0, "WOOL": 105.0, "FERTILIZER": 200.0,
}
TOWN_CENTER_PRODUCTS = frozenset(p for p in ECON_PRODUCTS if p != "FERTILIZER")
HARVEST_ECON_DIM = 34


def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _norm_roi(roi: float) -> float:
    return float(max(-1.0, min(1.0, math.log(max(1e-4, roi)) / 3.0)))


def _future_event_count(step: int, horizon_steps: int, period: int) -> int:
    step = int(step)
    horizon_steps = max(0, int(horizon_steps))
    period = max(1, int(period))
    return max(
        0,
        (step + horizon_steps) // period - step // period,
    )


def _future_sink_for_horizon(
    demand,
    *,
    step,
    horizon_steps,
    turns_per_day,
):
    shop_events = _future_event_count(step, horizon_steps, 4)
    center_events = _future_event_count(
        step, horizon_steps, turns_per_day
    )
    return {
        product: (
            float(demand.get(product, 0.0)) * float(shop_events)
            + (
                float(center_events)
                if product in TOWN_CENTER_PRODUCTS
                else 0.0
            )
        )
        for product in ECON_PRODUCTS
    }


def _sequential_sale_revenue(
    product: str,
    units: float,
    inventory: float,
):
    quantity = max(0, int(round(float(units))))
    market_inventory = max(0, int(round(float(inventory))))
    revenue = 0.0
    first_price = float(market_price(product, market_inventory))
    for _ in range(quantity):
        price = float(market_price(product, market_inventory))
        revenue += price
        if price > 1.0:
            market_inventory += 1
    average_price = (
        revenue / max(1, quantity)
        if quantity > 0
        else first_price
    )
    return {
        "units": quantity,
        "revenue": float(revenue),
        "first_price": float(first_price),
        "average_price": float(average_price),
        "post_inventory": int(market_inventory),
    }


def _time_state(observation, configuration=None):
    turns_per_day = max(1, int(_get(configuration, "turnsPerDay", 24) or 24))
    episode_steps = int(_get(configuration, "episodeSteps", 720) or 720)
    step = int(_get(observation, "step", 0) or 0)
    # Some observations expose day explicitly; otherwise derive it exactly.
    day = int(_get(observation, "day", step // turns_per_day) or 0)
    remaining_steps = max(0, episode_steps - 1 - step)
    return step, day, turns_per_day, episode_steps, remaining_steps, remaining_steps / float(turns_per_day)


def _new_crop_units(crop: str, remaining_days: float, harvest_buffer: float = 0.25) -> float:
    cd = CROP_META[crop]
    if remaining_days < cd["first"] + harvest_buffer:
        return 0.0
    if cd["ongoing"]:
        cycles = 1 + math.floor((remaining_days - cd["first"]) / max(1.0, cd["interval"]))
        return float(max(0.0, min(cd["max_yield"], cycles)))
    window_start = float((int(cd["max_day"]) + 1) // 2)
    latest_day = min(float(cd["max_day"]), remaining_days - harvest_buffer)
    water_days = max(0, math.floor(latest_day - window_start) + 1)
    return float(min(cd["max_yield"], 1.0 + water_days))


def _existing_crop_units(tile: dict, current_day: int, remaining_days: float) -> float:
    crop = str(tile.get("crop", ""))
    if crop not in CROP_META:
        return 0.0
    cd = CROP_META[crop]
    planted = int(tile.get("planted_day", current_day) or current_day)
    age = max(0, current_day - planted)
    current = max(0.0, float(tile.get("yield_units", 0) or 0))
    end_age = age + max(0, int(math.floor(remaining_days)))
    if end_age < cd["first"]:
        return 0.0
    if cd["ongoing"]:
        future = 0
        for next_age in range(age + 1, end_age + 1):
            dsf = next_age - int(cd["first"])
            if dsf < 0 or dsf % max(1, int(cd["interval"])) != 0:
                continue
            production_count = dsf // max(1, int(cd["interval"])) + 1
            if production_count <= int(cd["max_yield"]):
                future += 1
        return current + float(future)
    if age > cd["max_day"] + 1:
        return current
    window_start = int((int(cd["max_day"]) + 1) // 2)
    latest_age = min(int(cd["max_day"]), end_age)
    future_water_days = max(0, latest_age - max(age + 1, window_start) + 1)
    return min(float(cd["max_yield"]), current + float(future_water_days))


def _existing_animal_units(tile: dict, current_day: int, remaining_days: float) -> float:
    animal = str(tile.get("animal", ""))
    if animal not in ANIMAL_META:
        return 0.0
    ad = ANIMAL_META[animal]
    placed = int(tile.get("placed_day", current_day) or current_day)
    age = max(0, current_day - placed)
    end_age = age + max(0, int(math.floor(remaining_days)))
    current = max(0.0, float(tile.get("yield_units", 0) or 0))
    future = 0
    for next_age in range(age + 1, end_age + 1):
        dsf = next_age - int(ad["first"])
        if dsf >= 0 and dsf % max(1, int(ad["interval"])) == 0:
            future += 1
    return current + float(future)


def _farm_stats(farm, current_day: int, remaining_days: float):
    stats = {
        "empty": 0.0, "weed": 0.0, "plants": 0.0, "animals": 0.0,
        "coop_empty": 0.0, "pasture_empty": 0.0, "harvestable_free_soon": 0.0,
        "crop_counts": {x: 0.0 for x in CROP_META},
        "animal_counts": {x: 0.0 for x in ANIMAL_META},
        "projected": {x: 0.0 for x in ECON_PRODUCTS},
    }
    for row in list(_get(farm, "tiles", []) or []):
        for tile in row or []:
            if tile is None:
                stats["empty"] += 1.0
                continue
            if tile == "LOCKED" or not isinstance(tile, dict):
                continue
            kind = str(tile.get("kind", "") or "")
            if kind == "WEED":
                stats["weed"] += 1.0
            crop = tile.get("crop")
            animal = tile.get("animal")
            if kind == "PLANT" and crop in CROP_META:
                stats["plants"] += 1.0
                stats["crop_counts"][crop] += 1.0
                stats["projected"][crop] += _existing_crop_units(tile, current_day, remaining_days)
                cd = CROP_META[crop]
                age = current_day - int(tile.get("planted_day", current_day) or current_day)
                if not cd["ongoing"] and age >= cd["first"] and float(tile.get("yield_units", 0) or 0) > 0:
                    stats["harvestable_free_soon"] += 1.0
            if animal in ANIMAL_META:
                stats["animals"] += 1.0
                stats["animal_counts"][animal] += 1.0
                product = ANIMAL_META[animal]["product"]
                stats["projected"][product] += _existing_animal_units(tile, current_day, remaining_days)
            elif kind == "COOP":
                stats["coop_empty"] += 1.0
            elif kind == "PASTURE":
                stats["pasture_empty"] += 1.0
    return stats


def projected_farm_output(
    observation,
    horizon_days,
    *,
    player=None,
    configuration=None,
):
    _, current_day, _, _, _, remaining_days = _time_state(
        observation, configuration
    )
    farms = list(_get(observation, "farms", []) or [])
    if player is None:
        player = int(_get(observation, "player", 0) or 0)
    player = int(player)
    if not (0 <= player < len(farms)):
        return {product: 0.0 for product in ECON_PRODUCTS}
    horizon = max(0.0, min(float(horizon_days), float(remaining_days)))
    return dict(
        _farm_stats(farms[player], current_day, horizon)["projected"]
    )


def _projected_crop_sale(
    crop,
    *,
    units,
    own,
    rival,
    current_day,
    remaining_days,
    step,
    turns_per_day,
    remaining_steps,
    demand,
    inventory,
    horizon_extra_days=0.0,
):
    cd = CROP_META[crop]
    sale_horizon_days = min(
        float(remaining_days),
        max(
            0.0,
            float(cd["first"]) + max(0.0, float(horizon_extra_days)),
        ),
    )
    horizon_steps = min(
        int(remaining_steps),
        max(
            0,
            int(math.ceil(sale_horizon_days * turns_per_day)),
        ),
    )
    horizon_days = horizon_steps / float(max(1, turns_per_day))
    own_h = _farm_stats(own, current_day, horizon_days)
    rival_h = _farm_stats(rival, current_day, horizon_days)
    sink = _future_sink_for_horizon(
        demand,
        step=step,
        horizon_steps=horizon_steps,
        turns_per_day=turns_per_day,
    )
    projected_inventory = (
        float(_get(inventory, crop, 10000.0) or 0.0)
        + float(own_h["projected"].get(crop, 0.0))
        + float(rival_h["projected"].get(crop, 0.0))
        - float(sink.get(crop, 0.0))
    )
    sale = _sequential_sale_revenue(
        crop,
        units,
        max(0.0, projected_inventory),
    )
    sale.update(
        {
            "horizon_steps": int(horizon_steps),
            "horizon_days": float(horizon_days),
            "projected_inventory": float(max(0.0, projected_inventory)),
            "future_sink": float(sink.get(crop, 0.0)),
            "background_supply": float(
                own_h["projected"].get(crop, 0.0)
                + rival_h["projected"].get(crop, 0.0)
            ),
        }
    )
    return sale


def strategy_snapshot(
    observation,
    configuration=None,
    *,
    projected_crop_value=False,
    projected_crop_horizon_extra_days=0.0,
):
    step, current_day, turns_per_day, episode_steps, remaining_steps, remaining_days = _time_state(observation, configuration)
    town = _get(observation, "town", {}) or {}
    unlocked = list(_get(town, "unlocked_shops", []) or [])
    demand = {p: 0.0 for p in ECON_PRODUCTS}
    for shop in unlocked:
        products = ECON_SHOP_PRODUCTS.get(str(shop), ())
        mult = 2.0 if len(products) == 1 else 1.0
        for product in products:
            if product in demand:
                demand[product] += mult

    market = _get(observation, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    prices = _get(market, "prices", {}) or {}
    farms = list(_get(observation, "farms", []) or [])
    player = int(_get(observation, "player", 0) or 0)
    own = farms[player] if farms and 0 <= player < len(farms) else {}
    rival = farms[1 - player] if len(farms) >= 2 and 0 <= player < 2 else {}
    own_s = _farm_stats(own, current_day, remaining_days)
    rival_s = _farm_stats(rival, current_day, remaining_days)

    workers = 1.0 + len(list(_get(own, "hands", []) or []))
    # Movement, pickup/place and market travel consume most raw actions. Reserve
    # 45% of theoretical actions for mandatory plant/animal maintenance.
    daily_care_budget = max(1.0, workers * turns_per_day * 0.45)
    mandatory_load = 1.35 * own_s["plants"] + 1.60 * own_s["animals"]

    def care_factor(extra_load: float):
        return float(max(0.12, min(1.0, daily_care_budget / max(1.0, mandatory_load + extra_load))))

    if own_s["empty"] > 0:
        crop_space = 1.0
    elif own_s["harvestable_free_soon"] > 0:
        crop_space = 0.75
    elif own_s["weed"] > 0:
        crop_space = 0.45
    else:
        crop_space = 0.05

    def animal_space(structure: str):
        free_struct = own_s["coop_empty"] if structure == "COOP" else own_s["pasture_empty"]
        if free_struct > 0:
            return 1.0
        if own_s["empty"] > 0:
            return 0.82
        if own_s["weed"] > 0 or own_s["harvestable_free_soon"] > 0:
            return 0.45
        return 0.05

    # Exact town sink cadence from the bundled engine: each unlocked shop
    # consumes every 4 steps; town center consumes all non-fertilizer products
    # every 24 steps. This offsets projected player/rival supply before glut.
    shop_events = max(0, remaining_steps // 4)
    center_events = max(0, remaining_steps // 24)
    future_sink = {
        p: demand[p] * shop_events + (center_events if p in TOWN_CENTER_PRODUCTS else 0.0)
        for p in ECON_PRODUCTS
    }

    def glut_factor(product: str, added_units: float):
        current_inv = float(_get(inventory, product, 10000.0) or 0.0)
        projected = own_s["projected"][product] + rival_s["projected"][product] + max(0.0, added_units)
        excess = max(0.0, current_inv - 10000.0 + projected - future_sink[product])
        saturation = _clip01(excess / max(1.0, MARKET_CAPACITY[product]))
        return float(max(0.25, 1.0 - 0.70 * saturation))

    def timing_factor(first_yield: float):
        if remaining_days < first_yield + 0.25:
            return 0.0
        turnover = math.sqrt(min(1.0, 4.0 / max(1.0, first_yield)))
        slack = _clip01((remaining_days - first_yield) / max(1.0, first_yield))
        slack = 0.35 + 0.65 * slack
        return float(turnover * slack)

    crop_roi = {}
    crop_score = {}
    crop_viable = {}
    crop_projected_price = {}
    crop_projected_revenue = {}
    crop_projected_inventory = {}
    crop_projected_horizon = {}
    crop_undersupply_ratio = {}
    for crop, cd in CROP_META.items():
        units = _new_crop_units(crop, remaining_days)
        viable = units > 0.0 and crop_space > 0.10 and care_factor(1.35) > 0.20
        crop_viable[crop] = bool(viable)
        projected = None
        if not viable:
            roi = 1e-4
        elif projected_crop_value:
            projected = _projected_crop_sale(
                crop,
                units=units,
                own=own,
                rival=rival,
                current_day=current_day,
                remaining_days=remaining_days,
                step=step,
                turns_per_day=turns_per_day,
                remaining_steps=remaining_steps,
                demand=demand,
                inventory=inventory,
                horizon_extra_days=projected_crop_horizon_extra_days,
            )
            revenue = float(projected["revenue"])
            roi = (
                revenue
                * care_factor(1.35)
                * crop_space
                * timing_factor(cd["first"])
                / max(1.0, cd["seed"])
            )
            roi = max(1e-4, roi)
        else:
            price = float(_get(prices, crop, ECON_BASE_PRICE[crop]) or 0.0)
            demand_boost = 1.0 + 0.12 * demand[crop]
            revenue = units * price * demand_boost * glut_factor(crop, units)
            roi = revenue * care_factor(1.35) * crop_space * timing_factor(cd["first"]) / max(1.0, cd["seed"])
            roi = max(1e-4, roi)
        crop_roi[crop] = float(roi)
        crop_score[crop] = _norm_roi(roi)
        if projected is not None:
            crop_projected_price[crop] = float(projected["average_price"])
            crop_projected_revenue[crop] = float(projected["revenue"])
            crop_projected_inventory[crop] = float(
                projected["projected_inventory"]
            )
            crop_projected_horizon[crop] = int(projected["horizon_steps"])
            crop_undersupply_ratio[crop] = max(
                0.0,
                (
                    10000.0
                    - float(projected["projected_inventory"])
                )
                / max(1.0, float(MARKET_CAPACITY[crop])),
            )

    wheat_price = float(_get(prices, "WHEAT", ECON_BASE_PRICE["WHEAT"]) or ECON_BASE_PRICE["WHEAT"])
    wheat_feed_shadow = max(3.0, 0.25 * wheat_price)
    animal_roi = {}
    animal_score = {}
    animal_viable = {}
    for animal, ad in ANIMAL_META.items():
        viable = remaining_days >= ad["first"] + 0.25 and animal_space(ad["structure"]) > 0.10 and care_factor(1.60) > 0.20
        animal_viable[animal] = bool(viable)
        if not viable:
            roi = 1e-4
        else:
            cycles = 1.0 + math.floor((remaining_days - ad["first"]) / max(1.0, ad["interval"]))
            product = ad["product"]
            price = float(_get(prices, product, ECON_BASE_PRICE[product]) or 0.0)
            demand_boost = 1.0 + 0.12 * demand[product]
            care_bonus = 1.0 + 0.20 * _clip01((daily_care_budget - mandatory_load) / daily_care_budget)
            revenue = cycles * price * demand_boost * glut_factor(product, cycles)
            total_cost = ad["cost"] + remaining_days * wheat_feed_shadow
            roi = revenue * care_factor(1.60) * care_bonus * animal_space(ad["structure"]) * timing_factor(ad["first"]) / max(1.0, total_cost)
            roi = max(1e-4, roi)
        animal_roi[animal] = float(roi)
        animal_score[animal] = _norm_roi(roi)

    pressure = {p: float(rival_s["projected"].get(p, 0.0)) for p in ECON_PRODUCTS}
    return {
        "step": int(step),
        "remaining_steps": int(remaining_steps),
        "remaining_days": float(remaining_days),
        "demand": demand,
        "future_sink": future_sink,
        "rival_pressure": pressure,
        "crop_roi": crop_roi,
        "crop_score": crop_score,
        "crop_viable": crop_viable,
        "crop_projected_price": crop_projected_price,
        "crop_projected_revenue": crop_projected_revenue,
        "crop_projected_inventory": crop_projected_inventory,
        "crop_projected_horizon": crop_projected_horizon,
        "crop_undersupply_ratio": crop_undersupply_ratio,
        "animal_roi": animal_roi,
        "animal_score": animal_score,
        "animal_viable": animal_viable,
        "care_budget": float(daily_care_budget),
        "care_load": float(mandatory_load),
        "care_headroom": float(max(0.0, daily_care_budget - mandatory_load)),
        "crop_space_factor": float(crop_space),
        "own_projected": dict(own_s["projected"]),
        "rival_projected": dict(rival_s["projected"]),
    }


def harvest_context_features(observation, configuration=None):
    """34D checkpoint-safe V4.6 route economics.

    Layout:
      0:8   harvest-feasible ROI scores (3 animal, 5 crop)
      8:16  hard feasibility gates
      16:24 time-to-first-yield slack
      24:32 rival projected supply net of future town demand
      32    maintenance headroom ratio
      33    crop-space factor

    The branch consuming this vector is zero-residual at migration, so old
    checkpoints preserve behavior until PPO/BC learns to use these features.
    """
    snap = strategy_snapshot(observation, configuration)
    animal_order = ("GOOSE", "COW", "SHEEP")
    crop_order = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")

    values = [float(snap["animal_score"][x]) for x in animal_order]
    values += [float(snap["crop_score"][x]) for x in crop_order]
    values += [1.0 if snap["animal_viable"][x] else 0.0 for x in animal_order]
    values += [1.0 if snap["crop_viable"][x] else 0.0 for x in crop_order]

    remaining_days = float(snap["remaining_days"])
    for animal in animal_order:
        first = float(ANIMAL_META[animal]["first"])
        values.append(float(max(-1.0, min(1.0, (remaining_days - first) / max(1.0, first)))))
    for crop in crop_order:
        first = float(CROP_META[crop]["first"])
        values.append(float(max(-1.0, min(1.0, (remaining_days - first) / max(1.0, first)))))

    for animal in animal_order:
        product = ANIMAL_META[animal]["product"]
        net = float(snap["rival_projected"].get(product, 0.0)) - float(snap["future_sink"].get(product, 0.0))
        cap = max(1.0, float(MARKET_CAPACITY[product]))
        values.append(float(max(-1.0, min(1.0, net / cap))))
    for crop in crop_order:
        net = float(snap["rival_projected"].get(crop, 0.0)) - float(snap["future_sink"].get(crop, 0.0))
        cap = max(1.0, float(MARKET_CAPACITY[crop]))
        values.append(float(max(-1.0, min(1.0, net / cap))))

    care_budget = max(1e-6, float(snap["care_budget"]))
    values.append(float(max(0.0, min(1.0, float(snap["care_headroom"]) / care_budget))))
    values.append(float(max(0.0, min(1.0, float(snap["crop_space_factor"])))))

    if len(values) != HARVEST_ECON_DIM:
        raise RuntimeError(f"harvest economic width mismatch: {len(values)} != {HARVEST_ECON_DIM}")
    return values
