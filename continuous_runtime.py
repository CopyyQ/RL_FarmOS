from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from kaggrl.v4_option_model import V4OptionPolicy as _BaseV4OptionPolicy
from kaggrl.clock import resolve_clock
from kaggrl.macro_policy import MacroPolicy
from kaggrl.observation import ObservationEncoder
from kaggrl.v45_macro_data import load_v45_macro_data
from kaggrl.v4_market_race import front_run_future_sales
from kaggrl.v4_options import MARKET_MODES, V4Option, compile_market_mode
from kaggrl.v4_structural_gate import structural_override_for_state
from kaggrl.v45_economics import HARVEST_ECON_DIM, harvest_context_features
from kaggrl.v4_farm_supervisor import (
    farm_survival_risk,
    MICRO_TASKS,
    MICRO_LOCAL_DIM,
    PLANT_CONTEXT_DIM,
    safe_idle_hand_indices,
    micro_local_features,
    compile_micro_task,
    state_aware_market_orders,
)


class V4OptionPolicy(_BaseV4OptionPolicy):
    """Original V4 policy plus lightweight hidden-state helpers used by RL runtime."""

    def encode(self, obs, clock, state=None):
        z = self.input_proj(obs)
        z = torch.tanh(z + self.clock_proj(clock.to(z.dtype)))
        return self.lstm(z, state)

    def route_logits_from_hidden(self, hidden, clock):
        return self.route_head(hidden) + self.route_clock_head(clock)

    def market_logits_from_hidden(self, hidden, clock):
        return self.market_head(hidden) + self.market_clock_head(clock)


HORIZONS = (1, 12, 24, 48, 72, 96)

# V4.3 economic/shop context. These features are deliberately outside the
# frozen parent encoder so V4.2 checkpoints remain loadable. The learned
# economic branch starts as an exact zero residual and can adapt via PPO.
ECONOMIC_DIM = 48
OPPONENT_DIM = 24
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
ECON_ANIMALS = {
    "GOOSE": (300.0, 4.0, 1.0, "EGG"),
    "COW": (400.0, 8.0, 2.0, "MILK"),
    "SHEEP": (500.0, 6.0, 3.0, "WOOL"),
}
ECON_CROPS = {
    "WHEAT": (10.0, 2.0, 6.0),
    "CARROT": (20.0, 2.0, 4.0),
    "TOMATO": (50.0, 8.0, 4.0),
    "STRAWBERRY": (100.0, 10.0, 4.0),
    "MELON": (80.0, 10.0, 6.0),
}


# Frozen V4.3/V4.5 feature semantics. Do not change these without a new
# zero-residual branch; promoted checkpoints were trained against this scale.
LEGACY_ECON_ANIMALS = {
    "GOOSE": (300.0, 4.0, 1.0, "EGG"),
    "COW": (400.0, 8.0, 2.0, "MILK"),
    "SHEEP": (500.0, 6.0, 3.0, "WOOL"),
}
LEGACY_ECON_CROPS = {
    "WHEAT": (10.0, 2.0, 6.0),
    "CARROT": (20.0, 2.0, 4.0),
    "TOMATO": (50.0, 8.0, 4.0),
    "STRAWBERRY": (100.0, 10.0, 4.0),
    "MELON": (80.0, 10.0, 6.0),
}

def _econ_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def economic_context_features(observation, configuration=None):
    """Return 48 normalized economic opportunity features for V4.3."""
    town = _econ_get(observation, "town", {}) or {}
    unlocked = list(_econ_get(town, "unlocked_shops", []) or [])
    shop_counts = {shop: unlocked.count(shop) for shop in ECON_SHOPS}

    demand = {item: 0.0 for item in ECON_PRODUCTS}
    for shop in unlocked:
        products = ECON_SHOP_PRODUCTS.get(str(shop), ())
        multiplier = 2.0 if len(products) == 1 else 1.0
        for item in products:
            if item in demand:
                demand[item] += multiplier

    market = _econ_get(observation, "market", {}) or {}
    inventory = _econ_get(market, "inventory", {}) or {}
    prices = _econ_get(market, "prices", {}) or {}

    features = []
    # 0:8 shop multiplicity. Repeated shops matter because every instance
    # consumes independently in the exact engine.
    features.extend(min(1.0, shop_counts[s] / 8.0) for s in ECON_SHOPS)
    # 8:17 aggregate demand pressure induced by currently unlocked shops.
    features.extend(min(1.0, demand[p] / 8.0) for p in ECON_PRODUCTS)
    # 17:26 current price relative to its neutral/base price.
    for product in ECON_PRODUCTS:
        price = float(_econ_get(prices, product, ECON_BASE_PRICE[product]) or 0.0)
        ratio = price / max(1.0, ECON_BASE_PRICE[product])
        features.append(float(max(-1.0, min(1.0, (ratio - 1.0) / 2.0))))
    # 26:35 scarcity: positive means inventory below neutral I0=10000.
    for product in ECON_PRODUCTS:
        inv = float(_econ_get(inventory, product, 10000.0) or 0.0)
        features.append(float(max(-1.0, min(1.0, (10000.0 - inv) / 4000.0))))

    try:
        clock = resolve_clock(observation, configuration)
        step = int(clock.step)
    except Exception:
        step = int(_econ_get(observation, "step", 0) or 0)
    episode_steps = int(_econ_get(configuration, "episodeSteps", 720) or 720)
    remaining_fraction = max(0.0, min(1.0, (episode_steps - 1 - step) / max(1, episode_steps - 1)))
    # Kaggriculture's default season is 30 days (720 / 24).
    remaining_days = 30.0 * remaining_fraction

    # 35:38 approximate animal capital-efficiency scores. The score combines
    # time-to-first-yield, current product price and shop-induced demand.
    for animal in ("GOOSE", "COW", "SHEEP"):
        cost, first_yield, interval, product = LEGACY_ECON_ANIMALS[animal]
        if remaining_days < first_yield:
            yields = 0.0
        else:
            yields = 1.0 + math.floor((remaining_days - first_yield) / max(1.0, interval))
        price = float(_econ_get(prices, product, ECON_BASE_PRICE[product]) or 0.0)
        demand_boost = 1.0 + 0.20 * demand[product]
        roi = max(1e-4, (yields * price * demand_boost) / max(1.0, cost))
        features.append(float(max(-1.0, min(1.0, math.log(roi + 1e-6) / 3.0))))

    # 38:43 crop opportunity scores using seed cost, achievable yield and the
    # current demand/price regime. These are heuristics, not hard-coded actions.
    for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"):
        seed_cost, first_yield, max_yield = LEGACY_ECON_CROPS[crop]
        viability = max(0.0, min(1.0, remaining_days / max(1.0, first_yield)))
        price = float(_econ_get(prices, crop, ECON_BASE_PRICE[crop]) or 0.0)
        demand_boost = 1.0 + 0.20 * demand[crop]
        roi = max(1e-4, (max_yield * price * demand_boost * viability) / max(1.0, seed_cost))
        features.append(float(max(-1.0, min(1.0, math.log(roi + 1e-6) / 3.0))))

    farms = list(_econ_get(observation, "farms", []) or [])
    player = int(_econ_get(observation, "player", 0) or 0)
    own = farms[player] if farms and 0 <= player < len(farms) else {}
    money = max(0.0, float(_econ_get(own, "money", 0.0) or 0.0))
    animal_counts = {"GOOSE": 0.0, "COW": 0.0, "SHEEP": 0.0}
    for row in list(_econ_get(own, "tiles", []) or []):
        for tile in row:
            animal = _econ_get(tile, "animal", None) if tile not in (None, "LOCKED") else None
            if animal in animal_counts:
                animal_counts[animal] += 1.0

    # 43:48 global context: remaining time, capital, current animal exposure.
    features.append(float(remaining_fraction))
    features.append(float(min(1.0, math.log1p(money) / math.log1p(200000.0))))
    features.extend(min(1.0, animal_counts[a] / 25.0) for a in ("GOOSE", "COW", "SHEEP"))

    arr = np.asarray(features, dtype=np.float32)
    if arr.shape != (ECONOMIC_DIM,):
        raise RuntimeError(f"economic feature width mismatch: {arr.shape}")
    return arr


def opponent_context_features(observation):
    """Public rival-farm state used by the V4.4 strategy branch."""
    farms = list(_econ_get(observation, "farms", []) or [])
    player = int(_econ_get(observation, "player", 0) or 0)
    if len(farms) < 2 or not (0 <= player < len(farms)):
        return np.zeros(OPPONENT_DIM, dtype=np.float32)
    own = farms[player]
    rival = farms[1 - player]
    own_money = float(_econ_get(own, "money", 0.0) or 0.0)
    rival_money = float(_econ_get(rival, "money", 0.0) or 0.0)
    crop_names = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
    animal_names = ("GOOSE", "COW", "SHEEP")
    product_names = ECON_PRODUCTS
    crop_counts = {x: 0.0 for x in crop_names}
    animal_counts = {x: 0.0 for x in animal_names}
    pressure = {x: 0.0 for x in product_names}
    unlocked = 0
    total_yield = 0.0
    animal_product = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
    for row in list(_econ_get(rival, "tiles", []) or []):
        for tile in row:
            if tile != "LOCKED":
                unlocked += 1
            if not isinstance(tile, dict):
                continue
            crop = tile.get("crop")
            animal = tile.get("animal")
            y = max(0.0, float(tile.get("yield_units", 0) or 0))
            total_yield += y
            if crop in crop_counts:
                crop_counts[crop] += 1.0
                pressure[crop] += 0.5 + y
            if animal in animal_counts:
                animal_counts[animal] += 1.0
                product = animal_product[animal]
                pressure[product] += 0.75 + y
                pressure["FERTILIZER"] += 0.20
    hands = len(list(_econ_get(rival, "hands", []) or []))
    features = [
        min(1.0, math.log1p(max(0.0, rival_money)) / math.log1p(200000.0)),
        float(max(-1.0, min(1.0, (own_money - rival_money) / 50000.0))),
        min(1.0, hands / 12.0),
        min(1.0, unlocked / 100.0),
    ]
    features.extend(min(1.0, crop_counts[x] / 25.0) for x in crop_names)
    features.extend(min(1.0, animal_counts[x] / 25.0) for x in animal_names)
    pvals = [min(1.0, pressure[x] / 30.0) for x in product_names]
    features.extend(pvals)
    features.append(max(pvals) if pvals else 0.0)
    active = sum(v > 0.0 for v in list(crop_counts.values()) + list(animal_counts.values()))
    features.append(min(1.0, active / 8.0))
    features.append(min(1.0, total_yield / 100.0))
    arr = np.asarray(features, dtype=np.float32)
    if arr.shape != (OPPONENT_DIM,):
        raise RuntimeError(f"opponent feature width mismatch: {arr.shape}")
    return arr


class ContinuousActor(nn.Module):
    def __init__(self, hidden_dim: int, clock_dim: int, route_count: int, market_count: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.clock_dim = int(clock_dim)
        self.route_count = int(route_count)
        self.market_count = int(market_count)
        inp = hidden_dim + clock_dim + route_count
        self.trunk = nn.Sequential(
            nn.LayerNorm(inp),
            nn.Linear(inp, 384),
            nn.SiLU(),
            nn.Linear(384, 256),
            nn.SiLU(),
        )
        self.economic_router = nn.Sequential(
            nn.LayerNorm(ECONOMIC_DIM),
            nn.Linear(ECONOMIC_DIM, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
        )
        self.opponent_router = nn.Sequential(
            nn.LayerNorm(OPPONENT_DIM),
            nn.Linear(OPPONENT_DIM, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
        )
        # V4.6 route-only harvest economics. The final layer is zero-init so
        # migrating a V4.5 checkpoint is exactly behavior preserving.
        self.harvest_router = nn.Sequential(
            nn.LayerNorm(HARVEST_ECON_DIM),
            nn.Linear(HARVEST_ECON_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, route_count),
        )
        micro_in = (
            hidden_dim + clock_dim + MICRO_LOCAL_DIM
            + ECONOMIC_DIM + OPPONENT_DIM
        )
        self.micro_trunk = nn.Sequential(
            nn.LayerNorm(micro_in),
            nn.Linear(micro_in, 128),
            nn.SiLU(),
        )
        self.micro_plant_context = nn.Sequential(
            nn.LayerNorm(PLANT_CONTEXT_DIM),
            nn.Linear(PLANT_CONTEXT_DIM, 32),
            nn.SiLU(),
            nn.Linear(32, 128),
        )
        self.micro_task = nn.Linear(128, len(MICRO_TASKS))
        self.micro_value = nn.Linear(128, 1)
        self.route_residual = nn.Linear(256, route_count)
        self.market = nn.Linear(256, market_count)
        self.horizon = nn.Linear(256, len(HORIZONS))
        self.value = nn.Linear(256, 1)

        # V4.1: market is conditioned on the selected route, while horizon is
        # conditioned on both selected route and market mode.  The final layers
        # are initialized to zero, so old V4 checkpoints preserve their exact
        # behaviour until the new conditional heads learn.
        self.route_action_embedding = nn.Embedding(route_count, 32)
        self.market_action_embedding = nn.Embedding(market_count, 16)
        self.market_condition = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.SiLU(),
            nn.Linear(128, market_count),
        )
        self.horizon_condition = nn.Sequential(
            nn.Linear(256 + 32 + 16, 128),
            nn.SiLU(),
            nn.Linear(128, len(HORIZONS)),
        )

    def forward(
        self,
        hidden,
        clock,
        base_route,
        *,
        economic=None,
        opponent=None,
        harvest_economic=None,
        route_action=None,
        market_action=None,
    ):
        base_oh = torch.nn.functional.one_hot(
            base_route.long(), self.route_count
        ).to(hidden.dtype)
        x = torch.cat((hidden, clock.to(hidden.dtype), base_oh), dim=-1)
        z = self.trunk(x)
        if economic is not None:
            econ = economic.to(z.dtype)
            z = z + torch.tanh(self.economic_router(econ))
        if opponent is not None:
            opp = opponent.to(z.dtype)
            z = z + torch.tanh(self.opponent_router(opp))
        harvest_route_residual = torch.zeros(
            (z.shape[0], self.route_count), dtype=z.dtype, device=z.device
        )
        if harvest_economic is not None:
            harvest = harvest_economic.to(z.dtype)
            harvest_route_residual = self.harvest_router(harvest)

        market_logits = self.market(z)
        horizon_logits = self.horizon(z)
        route_emb = None
        if route_action is not None:
            route_emb = self.route_action_embedding(route_action.long())
            market_logits = market_logits + self.market_condition(
                torch.cat((z, route_emb.to(z.dtype)), dim=-1)
            )
        if route_emb is not None and market_action is not None:
            market_emb = self.market_action_embedding(market_action.long())
            horizon_logits = horizon_logits + self.horizon_condition(
                torch.cat(
                    (z, route_emb.to(z.dtype), market_emb.to(z.dtype)),
                    dim=-1,
                )
            )

        return {
            "route_residual": self.route_residual(z) + harvest_route_residual,
            "market": market_logits,
            "horizon": horizon_logits,
            "value": self.value(z).squeeze(-1),
        }

    def micro_forward(
        self, hidden, clock, local, economic=None, opponent=None, plant_context=None
    ):
        if economic is None:
            economic = torch.zeros(
                (hidden.shape[0], ECONOMIC_DIM),
                dtype=hidden.dtype,
                device=hidden.device,
            )
        if opponent is None:
            opponent = torch.zeros(
                (hidden.shape[0], OPPONENT_DIM),
                dtype=hidden.dtype,
                device=hidden.device,
            )
        x = torch.cat(
            (
                hidden,
                clock.to(hidden.dtype),
                local.to(hidden.dtype),
                economic.to(hidden.dtype),
                opponent.to(hidden.dtype),
            ),
            dim=-1,
        )
        z = self.micro_trunk(x)
        if plant_context is not None:
            pc = plant_context.to(z.dtype)
            z = z + torch.tanh(self.micro_plant_context(pc))
        return {
            "task": self.micro_task(z),
            "value": self.micro_value(z).squeeze(-1),
        }


def init_actor(actor: ContinuousActor):
    with torch.no_grad():
        actor.route_residual.weight.mul_(0.02)
        actor.route_residual.bias.zero_()
        actor.market.weight.mul_(0.02)
        actor.market.bias.zero_()
        actor.market.bias[0] = 1.25
        actor.horizon.weight.mul_(0.02)
        actor.horizon.bias.zero_()
        actor.horizon.bias[3] = 1.5
        actor.value.weight.mul_(0.05)
        actor.value.bias.zero_()

        # V4.1 migration must be bit-stable across fresh Python processes.
        # Older V4 checkpoints do not contain these conditional modules, so
        # leaving PyTorch's random constructor values here made exact eval vary
        # from process to process.  Use deterministic analytic initialization
        # for every new upstream parameter, then zero the final residual
        # projections so old V4 behavior is preserved exactly at migration.
        def deterministic_fill(tensor, scale, phase):
            values = torch.arange(
                tensor.numel(),
                device=tensor.device,
                dtype=torch.float32,
            )
            values = torch.sin(values * 0.017 + float(phase)) * float(scale)
            tensor.copy_(values.reshape(tensor.shape).to(tensor.dtype))

        deterministic_fill(
            actor.route_action_embedding.weight, 0.02, 0.11
        )
        deterministic_fill(
            actor.market_action_embedding.weight, 0.02, 0.37
        )
        deterministic_fill(
            actor.market_condition[0].weight, 0.01, 0.73
        )
        actor.market_condition[0].bias.zero_()
        deterministic_fill(
            actor.horizon_condition[0].weight, 0.01, 1.13
        )
        actor.horizon_condition[0].bias.zero_()
        deterministic_fill(
            actor.economic_router[1].weight, 0.01, 1.71
        )
        actor.economic_router[1].bias.zero_()
        actor.economic_router[3].weight.zero_()
        actor.economic_router[3].bias.zero_()
        deterministic_fill(
            actor.opponent_router[1].weight, 0.01, 2.17
        )
        actor.opponent_router[1].bias.zero_()
        # V4.4 opponent branch is neutral at migration; old behavior is exact.
        actor.opponent_router[3].weight.zero_()
        actor.opponent_router[3].bias.zero_()
        deterministic_fill(
            actor.harvest_router[1].weight, 0.01, 2.43
        )
        actor.harvest_router[1].bias.zero_()
        # New harvest-economics branch is an exact zero residual at migration.
        actor.harvest_router[3].weight.zero_()
        actor.harvest_router[3].bias.zero_()
        deterministic_fill(
            actor.micro_trunk[1].weight, 0.01, 2.61
        )
        actor.micro_trunk[1].bias.zero_()
        actor.micro_plant_context[0].weight.fill_(1.0)
        actor.micro_plant_context[0].bias.zero_()
        deterministic_fill(
            actor.micro_plant_context[1].weight, 0.01, 3.17
        )
        actor.micro_plant_context[1].bias.zero_()
        # Exact zero residual at migration: old checkpoints are bit-stable.
        actor.micro_plant_context[3].weight.zero_()
        actor.micro_plant_context[3].bias.zero_()
        # Micro policy starts as KEEP. Deterministic eval is bit-compatible;
        # stochastic rollouts can still explore care tasks for PPO learning.
        actor.micro_task.weight.zero_()
        actor.micro_task.bias.zero_()
        actor.micro_task.bias[0] = 4.5
        actor.micro_value.weight.zero_()
        actor.micro_value.bias.zero_()

        actor.market_condition[-1].weight.zero_()
        actor.market_condition[-1].bias.zero_()
        actor.horizon_condition[-1].weight.zero_()
        actor.horizon_condition[-1].bias.zero_()


def load_actor_state_compatible(actor: ContinuousActor, state_dict):
    """Load old/new V4 actor states, including safe row-wise head expansion."""
    current = actor.state_dict()
    matched = {}
    partial = []
    expandable = {"micro_task.weight", "micro_task.bias"}
    for key, value in state_dict.items():
        if key not in current:
            continue
        target = current[key]
        if target.shape == value.shape:
            matched[key] = value
            continue
        if (
            key in expandable
            and target.ndim == value.ndim
            and target.shape[0] >= value.shape[0]
            and (target.ndim == 1 or target.shape[1:] == value.shape[1:])
        ):
            merged = target.clone()
            merged[: value.shape[0]].copy_(value.to(merged.dtype))
            matched[key] = merged
            partial.append((key, tuple(value.shape), tuple(target.shape)))
    actor.load_state_dict(matched, strict=False)
    missing = sorted(set(current) - set(matched))
    unexpected = sorted(set(state_dict) - set(matched))
    return {
        "matched": len(matched),
        "missing": missing,
        "unexpected": unexpected,
        "partial": partial,
    }


def load_parent(parent_path, device="cpu"):
    payload = torch.load(parent_path, map_location=device, weights_only=False)
    model = V4OptionPolicy(
        input_dim=int(payload["input_dim"]),
        route_count=len(payload["route_ids"]),
        market_mode_count=len(payload["market_modes"]),
        hidden_dim=int(payload["hidden_dim"]),
        clock_dim=int(payload["clock_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return payload, model


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor):
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class ScriptedTrajectoryRuntime:
    """Replay an exact strategic option trace recorded by ContinuousRuntime."""

    def __init__(self, parent_path, records):
        self.payload, self.parent = load_parent(parent_path, device="cpu")
        self.route_ids = tuple(int(x) for x in self.payload["route_ids"])
        self.route_to_class = {
            route_id: index for index, route_id in enumerate(self.route_ids)
        }
        self.market_modes = tuple(self.payload["market_modes"])
        self.encoder = ObservationEncoder(clock_schema="v4")
        self.allowed_market_modes = frozenset(
            {
                "KEEP_ROUTE",
                "LIQUIDATE_SHED",
                "HOLD_SALES",
                "FRONT_RUN_1",
                "FRONT_RUN_9",
            }
        )
        routes, new_routes, old_routes = load_v45_macro_data()
        self.market_macro = MacroPolicy(routes, new_routes, old_routes)
        self.records_by_step = {
            int(r["step"]): dict(r) for r in records
        }
        self.reset()

    def reset(self):
        self.state = None
        self.last_step = None
        self.last_player = None
        self.active_route = None
        self.active_until = -1
        self.mismatches = []
        self.applied = []
        self.market_macro.reset()

    @staticmethod
    def _market_signature(action):
        return tuple(
            tuple(order) if isinstance(order, (list, tuple)) else (order,)
            for order in (action.get("market") or [])
        )

    def _market_relevant(
        self,
        observation: dict[str, Any],
        configuration,
        chosen_route: int,
    ) -> bool:
        try:
            base_action = self.market_macro.action_for_route(
                observation, int(chosen_route), configuration
            )
            base_sig = self._market_signature(base_action)
            for mode in ("LIQUIDATE_SHED", "HOLD_SALES"):
                changed = compile_market_mode(
                    observation, base_action, mode
                )
                if self._market_signature(changed) != base_sig:
                    return True
            for mode, horizon in (
                ("FRONT_RUN_1", 1),
                ("FRONT_RUN_9", 9),
            ):
                changed, debts = front_run_future_sales(
                    observation,
                    base_action,
                    self.market_macro,
                    int(chosen_route),
                    configuration,
                    horizon=horizon,
                    existing_debts=None,
                )
                if debts or self._market_signature(changed) != base_sig:
                    return True
            return False
        except Exception:
            return True

    def __call__(self, observation: dict[str, Any], configuration, context):
        clock = resolve_clock(observation, configuration)
        step = int(clock.step)
        player = int(observation.get("player", 0))
        if (
            step == 0
            or self.last_step is None
            or step <= self.last_step
            or (self.last_player is not None and player != self.last_player)
        ):
            self.reset()
        self.last_step = step
        self.last_player = player

        encoded = self.encoder.encode(observation, configuration).astype(
            np.float32, copy=False
        )
        clock_np = np.asarray(clock.features(), dtype=np.float32)
        obs_t = torch.from_numpy(encoded).view(1, 1, -1)
        clk_t = torch.from_numpy(clock_np).view(1, 1, -1)
        with torch.inference_mode():
            hidden, self.state = self.parent.encode(
                obs_t, clk_t, self.state
            )
            self.state = tuple(v.detach() for v in self.state)

        if self.active_route is not None:
            if step < self.active_until and self.active_route in context.route_ids:
                return V4Option(route_id=int(self.active_route)), 1.0
            self.active_route = None
            self.active_until = -1

        record = self.records_by_step.get(step)
        if record is None:
            return V4Option(), 1.0

        base_route = int(context.base_route_id)
        expected_base = int(record.get("base_route_id", base_route))
        if base_route != expected_base:
            self.mismatches.append(
                {
                    "step": step,
                    "kind": "base_route",
                    "expected": expected_base,
                    "actual": base_route,
                }
            )

        chosen_route = int(record["chosen_route_id"])
        if chosen_route not in context.route_ids:
            self.mismatches.append(
                {
                    "step": step,
                    "kind": "route_not_compatible",
                    "route": chosen_route,
                    "compatible": [int(x) for x in context.route_ids],
                }
            )
            chosen_route = base_route

        market_mode = str(record.get("market_mode", "KEEP_ROUTE"))
        horizon = int(record.get("horizon", 1))
        market_relevant = self._market_relevant(
            observation, configuration, chosen_route
        )
        horizon_relevant = chosen_route != base_route
        if horizon_relevant and horizon > 1:
            self.active_route = chosen_route
            self.active_until = step + horizon

        base_class = self.route_to_class.get(base_route, -1)
        route_class = self.route_to_class.get(chosen_route, base_class)
        try:
            market_action = self.market_modes.index(market_mode)
        except ValueError:
            market_action = self.market_modes.index("KEEP_ROUTE")
        try:
            horizon_action = HORIZONS.index(horizon)
        except ValueError:
            horizon_action = 0
        market_mask_bits = 0
        for index, mode in enumerate(self.market_modes):
            if (
                mode == "KEEP_ROUTE"
                or mode in self.allowed_market_modes
            ):
                market_mask_bits |= 1 << index

        self.applied.append(
            {
                "step": step,
                "route_id": chosen_route,
                "market_mode": market_mode,
                "market_relevant": bool(market_relevant),
                "horizon": horizon,
                "horizon_relevant": bool(horizon_relevant),
                "hidden_f16": hidden[:, 0].detach().cpu().numpy().astype(
                    np.float16, copy=False
                ).tobytes(),
                "clock_f16": clock_np.astype(
                    np.float16, copy=False
                ).tobytes(),
                "base_route_class": int(base_class),
                "route_action": int(route_class),
                "market_action": int(market_action),
                "horizon_action": int(horizon_action),
                "market_mask_bits": int(market_mask_bits),
            }
        )
        return V4Option(
            route_id=(None if chosen_route == base_route else chosen_route),
            market_mode=market_mode,
        ), 1.0


class ContinuousRuntime:
    def __init__(
        self,
        parent_path,
        snapshot_path,
        *,
        stochastic=True,
        temperature=1.15,
        residual_scale=1.0,
        base_keep_bias=2.0,
        decision_every=24,
        decision_start=24,
        decision_end=712,
        allowed_market_modes=None,
        enable_structural_route_gate=True,
        micro_decision_every=4,
        enable_micro_policy=True,
        skill_cutover_step=576,
        skill_keep_penalty=1.25,
        seed=0,
    ):
        self.device = torch.device("cpu")
        self.payload, self.parent = load_parent(parent_path, self.device)
        snap = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        self.route_ids = tuple(int(x) for x in self.payload["route_ids"])
        self.route_to_class = {r: i for i, r in enumerate(self.route_ids)}
        self.market_modes = tuple(self.payload["market_modes"])
        if self.market_modes != tuple(MARKET_MODES):
            raise RuntimeError("market schema mismatch")
        self.actor = ContinuousActor(
            int(self.payload["hidden_dim"]),
            int(self.payload["clock_dim"]),
            len(self.route_ids),
            len(self.market_modes),
        )
        init_actor(self.actor)
        self.actor_load_info = load_actor_state_compatible(
            self.actor, snap["actor_state"]
        )
        self.actor.eval()
        self.encoder = ObservationEncoder(clock_schema="v4")
        self.stochastic = bool(stochastic)
        self.temperature = max(0.05, float(temperature))
        self.residual_scale = float(residual_scale)
        self.base_keep_bias = float(base_keep_bias)
        self.decision_every = max(1, int(decision_every))
        self.decision_start = int(decision_start)
        self.decision_end = int(decision_end)
        self.allowed_market_modes = frozenset(
            self.market_modes if allowed_market_modes is None
            else [str(x) for x in allowed_market_modes]
        )
        self.enable_structural_route_gate = bool(
            enable_structural_route_gate
        )
        self.micro_decision_every = max(1, int(micro_decision_every))
        self.enable_micro_policy = bool(enable_micro_policy)
        self.skill_cutover_step = max(0, int(skill_cutover_step))
        self.skill_keep_penalty = max(0.0, float(skill_keep_penalty))
        routes, new_routes, old_routes = load_v45_macro_data()
        self.market_macro = MacroPolicy(routes, new_routes, old_routes)
        self.route_first_divergence = {}
        for left in self.route_ids:
            if left not in routes:
                continue
            for right in self.route_ids:
                if right not in routes:
                    continue
                a, b = routes[left], routes[right]
                limit = min(len(a), len(b))
                divergence = next(
                    (i for i in range(limit) if a[i] != b[i]),
                    limit,
                )
                self.route_first_divergence[(int(left), int(right))] = int(divergence)
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(int(seed))
        self.reset()

    def reset(self):
        self.state = None
        self.last_step = None
        self.last_player = None
        self.active_route = None
        self.active_until = -1
        self.current_route_id = None
        self.structural_route_id = None
        self.structural_until_step = -1
        self.last_shop_signature = None
        self.pending_shop_replan = False
        self.records = []
        self.micro_records = []
        self.micro_active = {}
        self.micro_cache = None
        self.skill_mode_started = False
        self.market_macro.reset()

    def _sample(self, logits):
        dist = Categorical(logits=logits)
        if self.stochastic:
            # Categorical.sample does not accept a generator. Manual multinomial keeps
            # each exact game reproducible from its worker seed.
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1, generator=self.rng).squeeze(-1)
        else:
            action = logits.argmax(dim=-1)
        return action, dist.log_prob(action), dist.entropy()

    def _decision_step(self, step):
        return (
            self.decision_start <= step <= self.decision_end
            and step % self.decision_every == 0
        )

    @staticmethod
    def _market_signature(action):
        return tuple(
            tuple(order) if isinstance(order, (list, tuple)) else (order,)
            for order in (action.get("market") or [])
        )

    def _market_relevant(
        self,
        observation: dict[str, Any],
        configuration,
        chosen_route: int,
    ) -> bool:
        """Whether any allowed market mode can change the executable action."""
        try:
            base_action = self.market_macro.action_for_route(
                observation, int(chosen_route), configuration
            )
            base_sig = self._market_signature(base_action)

            for mode in ("LIQUIDATE_SHED", "HOLD_SALES"):
                if mode not in self.allowed_market_modes:
                    continue
                changed = compile_market_mode(
                    observation, base_action, mode
                )
                if self._market_signature(changed) != base_sig:
                    return True

            for mode, horizon in (
                ("FRONT_RUN_1", 1),
                ("FRONT_RUN_9", 9),
            ):
                if mode not in self.allowed_market_modes:
                    continue
                changed, debts = front_run_future_sales(
                    observation,
                    base_action,
                    self.market_macro,
                    int(chosen_route),
                    configuration,
                    horizon=horizon,
                    existing_debts=None,
                )
                if debts or self._market_signature(changed) != base_sig:
                    return True
            return False
        except Exception:
            # Never hide a potentially meaningful market decision because of
            # a relevance-probe failure.
            return True

    def __call__(self, observation: dict[str, Any], configuration, context):
        clock = resolve_clock(observation, configuration)
        step = int(clock.step)
        player = int(observation.get("player", 0))
        if (
            step == 0
            or self.last_step is None
            or step <= self.last_step
            or (self.last_player is not None and player != self.last_player)
        ):
            self.reset()
        self.last_step = step
        self.last_player = player

        encoded = self.encoder.encode(observation, configuration).astype(
            np.float32, copy=False
        )
        economic_np = economic_context_features(
            observation, configuration
        ).astype(np.float32, copy=False)
        opponent_np = opponent_context_features(observation).astype(
            np.float32, copy=False
        )
        harvest_np = np.asarray(
            harvest_context_features(observation, configuration), dtype=np.float32
        )
        clock_np = np.asarray(clock.features(), dtype=np.float32)
        obs_t = torch.from_numpy(encoded).view(1, 1, -1)
        clk_t = torch.from_numpy(clock_np).view(1, 1, -1)
        economic_t = torch.from_numpy(economic_np).view(1, -1)
        opponent_t = torch.from_numpy(opponent_np).view(1, -1)
        harvest_t = torch.from_numpy(harvest_np).view(1, -1)
        shop_signature = tuple(
            int(round(float(x) * 8.0)) for x in economic_np[:8]
        )
        shop_changed = (
            self.last_shop_signature is not None
            and shop_signature != self.last_shop_signature
        )
        if shop_changed:
            self.pending_shop_replan = True
        self.last_shop_signature = shop_signature
        with torch.inference_mode():
            hidden, self.state = self.parent.encode(obs_t, clk_t, self.state)
            self.state = tuple(v.detach() for v in self.state)
        self.micro_cache = {
            "step": step,
            "hidden": hidden[:, 0].detach().clone(),
            "clock": clk_t[:, 0].detach().clone(),
            "economic": economic_t.detach().clone(),
            "opponent": opponent_t.detach().clone(),
            "harvest_economic": harvest_t.detach().clone(),
        }

        macro_base_route = int(context.base_route_id)
        if (
            self.current_route_id is None
            or int(self.current_route_id) not in context.route_ids
        ):
            self.current_route_id = int(macro_base_route)
        base_route = int(self.current_route_id)
        base_class = self.route_to_class.get(base_route)
        if base_class is None:
            self.current_route_id = int(macro_base_route)
            base_route = int(macro_base_route)
            base_class = self.route_to_class.get(base_route)
        if base_class is None:
            return V4Option(), 1.0

        # Mirror the deterministic structural gate inside the rollout runtime so
        # PPO credit matches the action that is actually executed by the hybrid
        # policy. While the structural option is active, actor route/horizon
        # choices are not behaviorally responsible for reward and must not
        # receive policy gradient.
        structural_started = False
        structural_active = False
        if self.enable_structural_route_gate:
            if (
                self.structural_route_id is not None
                and step >= self.structural_until_step
            ):
                self.structural_route_id = None
                self.structural_until_step = -1
            if self.structural_route_id is None:
                decision = structural_override_for_state(
                    step=step,
                    # Mirror V4HybridPolicy exactly: the causal structural
                    # gate is keyed to the original macro base route, not the
                    # learned persistent handoff route.
                    base_route_id=macro_base_route,
                )
                if decision is not None:
                    self.structural_route_id = int(decision.route_id)
                    self.structural_until_step = int(decision.until_step)
                    structural_started = True
                    # Deterministic structural control supersedes any learned
                    # carry-over route from an earlier horizon.
                    self.active_route = None
                    self.active_until = -1
            structural_active = (
                self.structural_route_id is not None
                and step < self.structural_until_step
            )

        # During the interior of the structural window the actor makes no
        # executable strategic decision: route is fixed and market is forced
        # to KEEP_ROUTE by the hybrid policy. Do not create a PPO record.
        if structural_active and not structural_started:
            return V4Option(), 1.0

        survival = farm_survival_risk(observation)
        force_shop_replan = bool(
            self.pending_shop_replan and int(survival.get("risk", 0)) == 0
        )
        if self.active_route is not None:
            if (
                step < self.active_until
                and int(self.current_route_id) in context.route_ids
                and not force_shop_replan
            ):
                return V4Option(
                    route_id=(
                        None
                        if int(self.current_route_id) == macro_base_route
                        else int(self.current_route_id)
                    )
                ), 1.0
            # Commitment expired; keep following the inherited route, but
            # allow a new compatible re-plan at the next decision point.
            self.active_route = None
            self.active_until = -1

        compatible_routes = []
        for route_id in context.route_ids:
            route_id = int(route_id)
            if route_id not in self.route_to_class:
                continue
            divergence = self.route_first_divergence.get(
                (int(base_route), route_id), -1
            )
            # Candidate may diverge at the current step, but everything before
            # this frame must match the route whose state we actually inherit.
            if route_id == int(base_route) or int(divergence) >= step:
                compatible_routes.append(route_id)
        compatible = [self.route_to_class[r] for r in compatible_routes]
        if (
            not structural_started
            and not self._decision_step(step)
            and not force_shop_replan
        ):
            return V4Option(
                route_id=(
                    None if int(base_route) == macro_base_route else int(base_route)
                )
            ), 1.0

        h = hidden[:, 0]
        c = clk_t[:, 0]
        base_t = torch.tensor([base_class], dtype=torch.long)
        with torch.inference_mode():
            route_out = self.actor(
                h,
                c,
                base_t,
                economic=economic_t,
                opponent=opponent_t,
                harvest_economic=harvest_t,
            )
            parent_logits = self.parent.route_logits_from_hidden(h, c)
            keep_bias = torch.nn.functional.one_hot(
                base_t, len(self.route_ids)
            ).to(parent_logits.dtype) * self.base_keep_bias
            route_logits = (
                parent_logits
                + keep_bias
                + self.residual_scale * route_out["route_residual"]
            ) / self.temperature

        route_mask = torch.zeros(
            (1, len(self.route_ids)), dtype=torch.bool
        )
        route_mask[0, compatible] = True

        if structural_started:
            chosen_route = int(self.structural_route_id)
            chosen_route_class = int(self.route_to_class[chosen_route])
            route_mask[0, chosen_route_class] = True
            route_action = torch.tensor(
                [chosen_route_class], dtype=torch.long
            )
            route_logp = torch.zeros((1,), dtype=torch.float32)
            route_entropy = torch.zeros((1,), dtype=torch.float32)
            route_relevant = False
        else:
            if len(compatible) <= 1:
                chosen_route = int(base_route)
                chosen_route_class = int(base_class)
                route_action = torch.tensor(
                    [chosen_route_class], dtype=torch.long
                )
                route_logp = torch.zeros((1,), dtype=torch.float32)
                route_entropy = torch.zeros((1,), dtype=torch.float32)
                route_relevant = False
            else:
                route_logits = _masked_logits(route_logits, route_mask)
                route_action, route_logp, route_entropy = self._sample(route_logits)
                chosen_route_class = int(route_action.item())
                chosen_route = int(self.route_ids[chosen_route_class])
                route_relevant = True

        allowed_market = [
            i for i, mode in enumerate(self.market_modes)
            if mode == "KEEP_ROUTE" or mode in self.allowed_market_modes
        ]
        market_mask = torch.zeros(
            (1, len(self.market_modes)), dtype=torch.bool
        )
        market_mask[0, allowed_market] = True
        market_relevant = self._market_relevant(
            observation, configuration, chosen_route
        )
        if market_relevant:
            with torch.inference_mode():
                market_out = self.actor(
                    h,
                    c,
                    base_t,
                    economic=economic_t,
                    opponent=opponent_t,
                    harvest_economic=harvest_t,
                    route_action=route_action,
                )
            market_logits = _masked_logits(
                market_out["market"] / self.temperature, market_mask
            )
            market_action, market_logp, market_entropy = self._sample(
                market_logits
            )
            market_mode = str(self.market_modes[int(market_action.item())])
        else:
            keep_market_class = self.market_modes.index("KEEP_ROUTE")
            market_action = torch.tensor([keep_market_class], dtype=torch.long)
            market_logp = torch.zeros((1,), dtype=torch.float32)
            market_entropy = torch.zeros((1,), dtype=torch.float32)
            market_mode = "KEEP_ROUTE"

        # Horizon is only a learned action when the actor itself chooses a
        # non-base route. The causal structural gate owns its 48-step duration,
        # so that fixed horizon must not receive policy gradient.
        if structural_started:
            horizon = int(self.structural_until_step - step)
            horizon_action = torch.tensor(
                [HORIZONS.index(horizon)], dtype=torch.long
            )
            horizon_logp = torch.zeros((1,), dtype=torch.float32)
            horizon_entropy = torch.zeros((1,), dtype=torch.float32)
            horizon_relevant = False
        else:
            horizon_relevant = chosen_route != base_route
            if horizon_relevant:
                with torch.inference_mode():
                    horizon_out = self.actor(
                        h,
                        c,
                        base_t,
                        economic=economic_t,
                        opponent=opponent_t,
                        harvest_economic=harvest_t,
                        route_action=route_action,
                        market_action=market_action,
                    )
                horizon_logits = horizon_out["horizon"] / self.temperature
                horizon_action, horizon_logp, horizon_entropy = self._sample(
                    horizon_logits
                )
                horizon = int(HORIZONS[int(horizon_action.item())])
            else:
                horizon_action = torch.zeros((1,), dtype=torch.long)
                horizon_logp = torch.zeros((1,), dtype=torch.float32)
                horizon_entropy = torch.zeros((1,), dtype=torch.float32)
                horizon = 1

        if not structural_started:
            # Learned route handoffs are persistent state inheritance. Horizon
            # is a minimum commitment before the next re-plan, not a TTL that
            # blindly jumps back to an incompatible old macro tape.
            self.current_route_id = int(chosen_route)
            if horizon_relevant and horizon > 1:
                self.active_route = int(chosen_route)
                self.active_until = step + horizon
            else:
                self.active_route = None
                self.active_until = -1

        total_logp = float(
            route_logp.item() + market_logp.item() + horizon_logp.item()
        )
        total_entropy = float(
            route_entropy.item() + market_entropy.item() + horizon_entropy.item()
        )
        route_mask_bits = 0
        for idx in compatible:
            route_mask_bits |= 1 << int(idx)
        if structural_started:
            route_mask_bits |= 1 << int(chosen_route_class)
        market_mask_bits = 0
        for idx in allowed_market:
            market_mask_bits |= 1 << int(idx)

        self.records.append(
            {
                "step": step,
                "hidden_f16": h[0].numpy().astype(np.float16).tobytes(),
                "clock_f16": clock_np.astype(np.float16).tobytes(),
                "economic_f16": economic_np.astype(np.float16).tobytes(),
                "opponent_f16": opponent_np.astype(np.float16).tobytes(),
                "harvest_f16": harvest_np.astype(np.float16).tobytes(),
                "base_route_class": int(base_class),
                "route_action": chosen_route_class,
                "route_relevant": bool(route_relevant),
                "market_action": int(market_action.item()),
                "market_relevant": bool(market_relevant),
                "horizon_action": int(horizon_action.item()),
                "horizon_relevant": bool(horizon_relevant),
                "route_mask_bits": int(route_mask_bits),
                "market_mask_bits": int(market_mask_bits),
                "old_logp": total_logp,
                "old_value": float(route_out["value"][0].item()),
                "entropy": total_entropy,
                "chosen_route_id": chosen_route,
                "base_route_id": base_route,
                "macro_base_route_id": macro_base_route,
                "handoff_candidates": len(compatible_routes),
                "market_mode": market_mode,
                "horizon": horizon,
                "shop_replan": bool(force_shop_replan and not structural_started),
            }
        )
        if force_shop_replan and not structural_started:
            self.pending_shop_replan = False

        return V4Option(
            route_id=(
                None if chosen_route == macro_base_route else chosen_route
            ),
            market_mode=market_mode,
        ), 1.0

    def micro_action(
        self,
        observation: dict[str, Any],
        configuration,
        selected_route: int,
        action: dict[str, Any],
    ):
        """Learned task policy for hands the macro route never uses again."""
        if not self.enable_micro_policy or self.micro_cache is None:
            return action, {"applied": False, "records": 0}
        step = int(resolve_clock(observation, configuration).step)
        if int(self.micro_cache.get("step", -1)) != step:
            return action, {"applied": False, "records": 0}
        farms = list(observation.get("farms") or [])
        player = int(observation.get("player", 0) or 0)
        if not (0 <= player < len(farms)):
            return action, {"applied": False, "records": 0}
        hand_count = len(list(farms[player].get("hands") or []))
        route = self.market_macro.routes.get(int(selected_route))
        turns_per_day = int(
            configuration.get("turnsPerDay", 24)
            if isinstance(configuration, dict)
            else getattr(configuration, "turnsPerDay", 24)
        )
        idle = safe_idle_hand_indices(
            route, step, hand_count, turns_per_day=turns_per_day
        )
        if not idle:
            return action, {"applied": False, "records": 0}

        out = {
            "farmer": list((action or {}).get("farmer") or ["PASS"]),
            "hands": [list(x) for x in ((action or {}).get("hands") or [])],
            "market": [list(x) for x in ((action or {}).get("market") or [])],
        }
        while len(out["hands"]) < hand_count:
            out["hands"].append(["PASS"])
        day = step // max(1, turns_per_day)
        h = self.micro_cache["hidden"]
        c = self.micro_cache["clock"]
        new_records = 0
        applied = 0
        task_counts = {}
        for hand_idx in idle:
            key = (int(day), int(hand_idx))
            active = self.micro_active.get(key)
            if active is None or step >= int(active.get("until", -1)):
                local_np = np.asarray(
                    micro_local_features(observation, hand_idx),
                    dtype=np.float32,
                )
                local_t = torch.from_numpy(local_np).view(1, -1)
                with torch.inference_mode():
                    mout = self.actor.micro_forward(
                        h,
                        c,
                        local_t,
                        economic=self.micro_cache["economic"],
                        opponent=self.micro_cache["opponent"],
                    )
                    logits = mout["task"] / self.temperature
                    task_action, logp, entropy = self._sample(logits)
                task_idx = int(task_action.item())
                task_name = str(MICRO_TASKS[task_idx])
                active = {
                    "task": task_name,
                    "task_idx": task_idx,
                    "until": step + self.micro_decision_every,
                }
                self.micro_active[key] = active
                self.micro_records.append(
                    {
                        "step": step,
                        "day": int(day),
                        "hand_idx": int(hand_idx),
                        "hidden_f16": h[0].numpy().astype(np.float16).tobytes(),
                        "clock_f16": c[0].numpy().astype(np.float16).tobytes(),
                        "local_f16": local_np.astype(np.float16).tobytes(),
                        "economic_f16": self.micro_cache["economic"][0].numpy().astype(np.float16).tobytes(),
                        "opponent_f16": self.micro_cache["opponent"][0].numpy().astype(np.float16).tobytes(),
                        "task_action": task_idx,
                        "old_logp": float(logp.item()),
                        "old_value": float(mout["value"][0].item()),
                        "entropy": float(entropy.item()),
                    }
                )
                new_records += 1
            task_name = str(active["task"])
            compiled, meta = compile_micro_task(
                observation, hand_idx, task_name
            )
            if task_name != "KEEP" and bool(meta.get("valid", False)):
                out["hands"][hand_idx] = list(compiled)
                applied += int(str(compiled[0]) != "PASS")
            task_counts[task_name] = task_counts.get(task_name, 0) + 1
        return out, {
            "applied": bool(applied),
            "idle_hands": len(idle),
            "overrides": int(applied),
            "records": int(new_records),
            "tasks": task_counts,
        }
