from __future__ import annotations

from kaggrl.v45_economics import ECON_BASE_PRICE, ECON_PRODUCTS
from kaggrl.v4_farm_supervisor import apply_portfolio_switch_overlay
from tools.diag.shop_space import enumerate_shop_multisets


def _observation(*, step=216, animal=False, wheat=0):
    inventory = {product: 10000 for product in ECON_PRODUCTS}
    inventory["TOMATO"] = 9400
    inventory["WHEAT"] = 10100
    inventory["STRAWBERRY"] = 10200
    prices = {
        product: int(ECON_BASE_PRICE[product])
        for product in ECON_PRODUCTS
    }
    tiles = [[None] * 3 for _ in range(3)]
    if animal:
        tiles[0][0] = {
            "kind": "PASTURE",
            "animal": "COW",
            "fed_today": False,
            "cared_today": False,
            "placed_day": 0,
            "yield_units": 0,
        }
    farm = {"money": 10000, "hands": [], "tiles": tiles}
    rival = {
        "money": 10000,
        "hands": [],
        "tiles": [[None] * 3 for _ in range(3)],
    }
    return {
        "step": int(step),
        "day": int(step) // 24,
        "player": 0,
        "town": {
            "unlocked_shops": [
                "PIZZA_SHOP",
                "PIZZA_SHOP",
                "FARMERS_MARKET",
            ]
        },
        "market": {"inventory": inventory, "prices": prices},
        "farms": [farm, rival],
        "private": {
            "shed": {"WHEAT": int(wheat)} if wheat else {},
            "seeds": {"WHEAT": 5, "TOMATO": 5},
            "inventories": [{}],
        },
    }


def _run(obs, action, **overrides):
    kwargs = {
        "crop_improvement_ratio": 1.10,
        "feed_reserve_days": 2.0,
        "activation_step": 72,
        "projected_horizon_extra_days": 4.0,
        "min_undersupply_ratio": 0.0,
        "min_shop_demand": 1.0,
        "source_mode": "any",
    }
    kwargs.update(overrides)
    return apply_portfolio_switch_overlay(obs, action, **kwargs)


def test_shop_space_probability_mass_and_counts():
    rows6 = enumerate_shop_multisets(6)
    rows8 = enumerate_shop_multisets(8)
    assert len(rows6) == 1716
    assert len(rows8) == 6435
    assert abs(sum(x["probability"] for x in rows6) - 1.0) < 1e-12
    assert abs(sum(x["probability"] for x in rows8) - 1.0) < 1e-12


def test_before_activation_is_exact_noop():
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["BUY_SEED", "WHEAT", 2]],
    }
    out, meta = _run(_observation(step=48), action)
    assert out is action
    assert not meta["applied"]
    assert meta["reason"] == "before_activation"


def test_pizza_scarcity_switches_seed_to_tomato():
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["BUY_SEED", "WHEAT", 2]],
    }
    out, meta = _run(_observation(), action)
    assert len(out["market"]) == len(action["market"])
    assert out["market"][0][0] == "BUY_SEED"
    assert out["market"][0][1] == "TOMATO"
    assert out["market"][0][2] == 2
    assert meta["seed_switches"] == 1
    assert meta["plant_switches"] == 0


def test_feed_shortage_protects_wheat():
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["BUY_SEED", "WHEAT", 2]],
    }
    out, meta = _run(_observation(animal=True, wheat=0), action)
    assert out["market"] == action["market"]
    assert meta["seed_switches"] == 0
    assert meta["feed_need"] > meta["feed_available"]


def test_switch_only_never_drops_action_slots():
    action = {
        "farmer": ["PLANT", "WHEAT"],
        "hands": [["PASS"], ["PLANT", "WHEAT"]],
        "market": [
            ["BUY_SEED", "WHEAT", 2],
            ["SELL", "CARROT", 1],
        ],
    }
    out, _ = _run(_observation(), action)
    assert len(out["hands"]) == len(action["hands"])
    assert len(out["market"]) == len(action["market"])
    assert out["market"][1] == action["market"][1]


if __name__ == "__main__":
    test_shop_space_probability_mass_and_counts()
    test_before_activation_is_exact_noop()
    test_pizza_scarcity_switches_seed_to_tomato()
    test_feed_shortage_protects_wheat()
    test_switch_only_never_drops_action_slots()
    print("ECONOMIC_PORTFOLIO_TESTS_OK")
