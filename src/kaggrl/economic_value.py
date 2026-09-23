from __future__ import annotations

from typing import Any

from .constants import PRODUCTS
from .v4_market_race import MARKET_PARAMS, market_price
from .v45_economics import (
    ANIMAL_META,
    CROP_META,
    ECON_BASE_PRICE,
    ECON_SHOP_PRODUCTS,
    TOWN_CENTER_PRODUCTS,
    _existing_animal_units,
    _existing_crop_units,
)


def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _time_state(observation, configuration=None):
    turns_per_day = max(
        1, int(_get(configuration, "turnsPerDay", 24) or 24)
    )
    episode_steps = int(
        _get(configuration, "episodeSteps", 720) or 720
    )
    step = int(_get(observation, "step", 0) or 0)
    day = int(_get(observation, "day", step // turns_per_day) or 0)
    remaining_steps = max(0, episode_steps - 1 - step)
    return step, day, turns_per_day, remaining_steps


def _own_and_rival(observation):
    farms = list(_get(observation, "farms", []) or [])
    player = int(_get(observation, "player", 0) or 0)
    if len(farms) < 2 or player not in (0, 1):
        return {}, {}
    return farms[player], farms[1 - player]


def _own_shed(farm):
    private = _get(farm, "private", {}) or {}
    shed = _get(private, "shed", {}) or {}
    return {
        product: max(0, int(_get(shed, product, 0) or 0))
        for product in PRODUCTS
    }


def _market_state(observation):
    market = _get(observation, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    prices = _get(market, "prices", {}) or {}
    params = _get(market, "params", None)
    return (
        {
            product: int(
                _get(
                    inventory,
                    product,
                    MARKET_PARAMS[product]["I0"],
                )
                or 0
            )
            for product in PRODUCTS
        },
        {
            product: int(
                _get(
                    prices,
                    product,
                    ECON_BASE_PRICE[product],
                )
                or ECON_BASE_PRICE[product]
            )
            for product in PRODUCTS
        },
        params,
    )


def shop_sink_units(observation, configuration=None, horizon_steps=72):
    """Exact town-consumption count over a future horizon.

    The bundled engine consumes unlocked-shop products every 4 steps and town
    center products every 24 steps. We count only future events strictly after
    the current observation step.
    """
    step, _, turns_per_day, remaining = _time_state(
        observation, configuration
    )
    horizon = max(0, min(int(horizon_steps), int(remaining)))
    town = _get(observation, "town", {}) or {}
    shops = list(_get(town, "unlocked_shops", []) or [])

    sink = {product: 0 for product in PRODUCTS}
    for offset in range(1, horizon + 1):
        future_step = step + offset
        if future_step % 4 == 0:
            for shop_name in shops:
                products = ECON_SHOP_PRODUCTS.get(str(shop_name), ())
                multiplier = 2 if len(products) == 1 else 1
                for product in products:
                    sink[product] += multiplier
        if future_step % turns_per_day == 0:
            for product in TOWN_CENTER_PRODUCTS:
                sink[product] += 1
    return sink


def projected_market_inventory(
    observation,
    configuration=None,
    horizon_steps=72,
    supply_delta=None,
):
    inventory, _, _ = _market_state(observation)
    sink = shop_sink_units(
        observation,
        configuration,
        horizon_steps=horizon_steps,
    )
    supply_delta = supply_delta or {}
    return {
        product: max(
            0,
            int(inventory[product])
            - int(sink.get(product, 0))
            + int(supply_delta.get(product, 0) or 0),
        )
        for product in PRODUCTS
    }


def liquidation_value_for_item(
    item,
    quantity,
    market_inventory,
    params=None,
):
    """Return exact sequential SELL revenue and post-sale inventory.

    Mirrors the engine: every sold unit is quoted from current market
    inventory; a sale at price 1 does not increase market supply.
    """
    quantity = max(0, int(quantity or 0))
    inventory = int(market_inventory)
    revenue = 0
    for _ in range(quantity):
        price = int(market_price(item, inventory, params))
        revenue += price
        if price > 1:
            inventory += 1
    return int(revenue), int(inventory)


def exact_liquidation_value(
    quantities,
    market_inventory,
    params=None,
):
    """Liquidate a product bundle exactly under current market mechanics."""
    quantities = quantities or {}
    state = {
        product: int(
            _get(
                market_inventory,
                product,
                MARKET_PARAMS[product]["I0"],
            )
            or 0
        )
        for product in PRODUCTS
    }
    revenue_by_product = {}
    total = 0
    for product in PRODUCTS:
        quantity = max(
            0, int(_get(quantities, product, 0) or 0)
        )
        revenue, state[product] = liquidation_value_for_item(
            product,
            quantity,
            state[product],
            params,
        )
        revenue_by_product[product] = int(revenue)
        total += int(revenue)
    return {
        "total": int(total),
        "by_product": revenue_by_product,
        "post_inventory": state,
    }


def visible_future_output(
    farm,
    current_day,
    remaining_days,
):
    """Estimate visible crop/animal output without using private rival data."""
    output = {product: 0.0 for product in PRODUCTS}
    for row in list(_get(farm, "tiles", []) or []):
        for tile in list(row or []):
            if not isinstance(tile, dict):
                continue
            crop = str(tile.get("crop", "") or "")
            animal = str(tile.get("animal", "") or "")
            if crop in CROP_META:
                output[crop] += _existing_crop_units(
                    tile,
                    int(current_day),
                    float(remaining_days),
                )
            if animal in ANIMAL_META:
                product = str(ANIMAL_META[animal]["product"])
                output[product] += _existing_animal_units(
                    tile,
                    int(current_day),
                    float(remaining_days),
                )
    return output


def _int_bundle(bundle):
    return {
        product: max(0, int(round(float(bundle.get(product, 0.0)))))
        for product in PRODUCTS
    }


def economic_value_snapshot(
    observation,
    configuration=None,
    horizon_steps=72,
):
    """Shadow economic-value estimate for diagnostics/training labels.

    Runtime-safe: own private shed is used, but rival private inventory is
    never accessed. Rival value is estimated only from public cash + visible
    crops/animals.
    """
    step, day, turns_per_day, remaining = _time_state(
        observation, configuration
    )
    horizon = max(0, min(int(horizon_steps), int(remaining)))
    horizon_days = horizon / float(turns_per_day)

    own, rival = _own_and_rival(observation)
    own_money = int(_get(own, "money", 0) or 0)
    rival_money = int(_get(rival, "money", 0) or 0)
    shed = _own_shed(own)
    market_inventory, market_prices, params = _market_state(observation)

    current_liquidation = exact_liquidation_value(
        shed,
        market_inventory,
        params,
    )
    sink = shop_sink_units(
        observation,
        configuration,
        horizon_steps=horizon,
    )
    projected_inventory = {
        product: max(
            0,
            int(market_inventory[product]) - int(sink[product]),
        )
        for product in PRODUCTS
    }

    own_output = visible_future_output(
        own,
        current_day=day,
        remaining_days=horizon_days,
    )
    rival_output = visible_future_output(
        rival,
        current_day=day,
        remaining_days=horizon_days,
    )

    own_future_bundle = {
        product: int(shed[product])
        + int(round(own_output[product]))
        for product in PRODUCTS
    }
    own_future_liquidation = exact_liquidation_value(
        own_future_bundle,
        projected_inventory,
        params,
    )
    rival_visible_liquidation = exact_liquidation_value(
        _int_bundle(rival_output),
        projected_inventory,
        params,
    )

    current_net_worth = own_money + current_liquidation["total"]
    own_horizon_value = own_money + own_future_liquidation["total"]
    rival_visible_horizon_value = (
        rival_money + rival_visible_liquidation["total"]
    )

    scarcity_ratio = {}
    for product in PRODUCTS:
        base = max(1.0, float(ECON_BASE_PRICE[product]))
        scarcity_ratio[product] = (
            float(market_prices[product]) / base
        )

    return {
        "step": int(step),
        "day": int(day),
        "horizon_steps": int(horizon),
        "own_cash": int(own_money),
        "rival_cash": int(rival_money),
        "cash_margin": int(own_money - rival_money),
        "own_shed": shed,
        "market_inventory": market_inventory,
        "market_prices": market_prices,
        "scarcity_ratio": scarcity_ratio,
        "shop_sink": sink,
        "projected_market_inventory": projected_inventory,
        "own_visible_output": own_output,
        "rival_visible_output": rival_output,
        "own_liquidation_now": int(current_liquidation["total"]),
        "own_liquidation_now_by_product": current_liquidation["by_product"],
        "own_current_net_worth": int(current_net_worth),
        "own_horizon_liquidation": int(
            own_future_liquidation["total"]
        ),
        "rival_visible_horizon_liquidation": int(
            rival_visible_liquidation["total"]
        ),
        "own_horizon_value": int(own_horizon_value),
        "rival_visible_horizon_value": int(
            rival_visible_horizon_value
        ),
        "visible_horizon_advantage": int(
            own_horizon_value - rival_visible_horizon_value
        ),
    }
