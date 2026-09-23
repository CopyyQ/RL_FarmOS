from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from kaggrl.economic_value import (
    economic_value_snapshot,
    exact_liquidation_value,
    liquidation_value_for_item,
    projected_market_inventory,
    shop_sink_units,
)
from kaggrl.v4_market_race import MARKET_PARAMS, market_price


def test_exact_liquidation_matches_unit_sequence():
    inventory = 10000
    expected = 0
    for _ in range(5):
        price = market_price("MILK", inventory)
        expected += price
        if price > 1:
            inventory += 1
    revenue, final_inventory = liquidation_value_for_item(
        "MILK", 5, 10000
    )
    assert revenue == expected
    assert final_inventory == inventory


def test_bundle_liquidation_is_product_separable():
    inv = {name: 10000 for name in MARKET_PARAMS}
    result = exact_liquidation_value(
        {"WHEAT": 4, "MILK": 3, "WOOL": 2},
        inv,
    )
    wheat, _ = liquidation_value_for_item("WHEAT", 4, 10000)
    milk, _ = liquidation_value_for_item("MILK", 3, 10000)
    wool, _ = liquidation_value_for_item("WOOL", 2, 10000)
    assert result["total"] == wheat + milk + wool


def test_shop_sink_matches_engine_cadence():
    obs = {
        "step": 575,
        "town": {"unlocked_shops": ["SMOOTHIE_SHOP"]},
    }
    cfg = {"turnsPerDay": 24, "episodeSteps": 720}
    sink = shop_sink_units(obs, cfg, horizon_steps=24)
    # Shop events: 576, 580, 584, 588, 592, 596 (6 events).
    # Town center also consumes once at step 576.
    assert sink["MILK"] == 7
    assert sink["STRAWBERRY"] == 7
    assert sink["WHEAT"] == 1
    assert sink["FERTILIZER"] == 0


def test_projected_inventory_never_negative():
    obs = {
        "step": 575,
        "town": {"unlocked_shops": ["YARN_STORE"]},
        "market": {
            "inventory": {name: 0 for name in MARKET_PARAMS},
            "prices": {name: MARKET_PARAMS[name]["base"] for name in MARKET_PARAMS},
        },
    }
    cfg = {"turnsPerDay": 24, "episodeSteps": 720}
    projected = projected_market_inventory(
        obs, cfg, horizon_steps=24
    )
    assert all(value >= 0 for value in projected.values())


def test_snapshot_uses_own_private_shed_but_not_rival_private():
    obs = {
        "step": 575,
        "player": 0,
        "town": {"unlocked_shops": ["SMOOTHIE_SHOP"]},
        "market": {
            "inventory": {name: 10000 for name in MARKET_PARAMS},
            "prices": {name: MARKET_PARAMS[name]["base"] for name in MARKET_PARAMS},
        },
        "farms": [
            {
                "money": 50000,
                "private": {
                    "shed": {"MILK": 10, "WOOL": 2},
                    "seeds": {},
                },
                "tiles": [],
            },
            {
                "money": 60000,
                # Deliberately omit private rival inventory. The estimator
                # must remain runtime-safe and use only visible rival assets.
                "tiles": [],
            },
        ],
    }
    cfg = {"turnsPerDay": 24, "episodeSteps": 720}
    snap = economic_value_snapshot(obs, cfg, horizon_steps=24)
    assert snap["own_cash"] == 50000
    assert snap["rival_cash"] == 60000
    assert snap["own_shed"]["MILK"] == 10
    assert snap["own_liquidation_now"] > 0
    assert snap["own_current_net_worth"] > snap["own_cash"]
    assert snap["rival_visible_horizon_liquidation"] == 0
    assert snap["horizon_steps"] == 24


if __name__ == "__main__":
    test_exact_liquidation_matches_unit_sequence()
    test_bundle_liquidation_is_product_separable()
    test_shop_sink_matches_engine_cadence()
    test_projected_inventory_never_negative()
    test_snapshot_uses_own_private_shed_but_not_rival_private()
    print("ECONOMIC_VALUE_TESTS_OK")
