from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kaggrl.v45_economics import (
    strategy_snapshot,
    projected_farm_output,
    CROP_META,
    ANIMAL_META,
    ECON_BASE_PRICE,
)

MOVES = frozenset({"NORTH", "SOUTH", "EAST", "WEST"})
MICRO_TASKS = (
    "KEEP", "WATER", "FEED", "CARE", "COLLECT_FERT",
    "FERTILIZE", "HARVEST", "STORE_SHED", "FETCH_WHEAT", "DIG_WEED",
    "PLANT_BEST", "DEPLOY_ANIMAL",
)
MICRO_LOCAL_DIM = 43
PLANT_CONTEXT_DIM = 12
TILE_OPS = frozenset({
    "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "CARE",
    "COLLECT_FERTILIZER", "PLACE",
})


@dataclass(frozen=True)
class FarmCareResult:
    applied: bool
    critical_plants: int
    critical_animals: int
    overrides: int
    water_orders: int
    feed_orders: int
    care_orders: int
    fertilize_orders: int
    wheat_rescue_buys: int
    unit_changes: tuple[tuple[int, tuple, tuple], ...]


@dataclass(frozen=True)
class IdleLaborResult:
    applied: bool
    idle_hands: int
    moved: int
    watered: int
    fed: int
    cared: int
    harvested: int
    collected_fertilizer: int
    fertilized: int
    pickups: int
    rescue_buys: int


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_action(raw):
    if isinstance(raw, list) and raw:
        return list(raw)
    return ["PASS"]


def _farm(observation):
    farms = list(_get(observation, "farms", []) or [])
    player = int(_get(observation, "player", 0) or 0)
    if not (0 <= player < len(farms)):
        return None, player
    return farms[player], player


def _tile_at(farm, pos):
    if farm is None or not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return None
    x, y = int(pos[0]), int(pos[1])
    tiles = _get(farm, "tiles", []) or []
    if not (0 <= y < len(tiles) and 0 <= x < len(tiles[y])):
        return None
    return tiles[y][x]


def _positions(farm):
    return [list(_get(farm, "farmer", [0, 0]))] + [
        list(x) for x in (_get(farm, "hands", []) or [])
    ]


def _private(observation):
    return _get(observation, "private", {}) or {}


def _inventories(observation, n_units):
    invs = list(_get(_private(observation), "inventories", []) or [])
    while len(invs) < n_units:
        invs.append({})
    return invs


def _shed(observation):
    return _get(_private(observation), "shed", {}) or {}


def _shed_access(board_size):
    half = board_size // 2
    return (
        (half - 1, half - 1), (half, half - 1),
        (half - 1, half), (half, half),
    )


def _distance(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _step_toward(pos, target):
    x, y = int(pos[0]), int(pos[1])
    tx, ty = int(target[0]), int(target[1])
    if x < tx: return ["EAST"]
    if x > tx: return ["WEST"]
    if y < ty: return ["SOUTH"]
    if y > ty: return ["NORTH"]
    return ["PASS"]


def _asset_targets(farm):
    critical_plants, due_plants = [], []
    critical_animals, due_animals = [], []
    for y, row in enumerate(_get(farm, "tiles", []) or []):
        for x, tile in enumerate(row or []):
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT" and not tile.get("watered_today", False):
                target = (x, y)
                due_plants.append(target)
                if int(tile.get("consecutive_unwatered", 0) or 0) >= 1:
                    critical_plants.append(target)
            if "animal" in tile and not tile.get("fed_today", False):
                target = (x, y)
                due_animals.append(target)
                if int(tile.get("consecutive_unfed", 0) or 0) >= 1:
                    critical_animals.append(target)
    return critical_plants, due_plants, critical_animals, due_animals


def farm_survival_risk(observation):
    farm, _ = _farm(observation)
    if farm is None:
        return {"critical_plants": 0, "critical_animals": 0, "risk": 0}
    cp, _, ca, _ = _asset_targets(farm)
    return {
        "critical_plants": len(cp),
        "critical_animals": len(ca),
        "risk": len(cp) + 2 * len(ca),
    }


def _set_unit_action(out, unit_idx, replacement):
    if unit_idx == 0:
        before = tuple(out["farmer"])
        out["farmer"] = list(replacement)
    else:
        before = tuple(out["hands"][unit_idx - 1])
        out["hands"][unit_idx - 1] = list(replacement)
    return before


def _nearest(pos, targets, claimed):
    available = [t for t in targets if t not in claimed]
    if not available:
        return None
    return min(available, key=lambda t: (_distance(pos, t), t[1], t[0]))


def supervise_farm_care(observation, action, *, remaining_steps=None):
    """Position-safe emergency repair for static macro choreography.

    Never moves a unit and never changes market orders. Only converts PASS on
    the tile the unit already occupies into a life-saving action. This keeps
    future macro positions intact while preventing avoidable asset loss.
    """
    out = {
        "farmer": _normalize_action((action or {}).get("farmer")),
        "hands": [_normalize_action(x) for x in ((action or {}).get("hands") or [])],
        "market": [list(x) for x in ((action or {}).get("market") or []) if isinstance(x, (list, tuple))],
    }
    farm, _ = _farm(observation)
    if farm is None:
        return out, FarmCareResult(False, 0, 0, 0, 0, 0, 0, 0, 0, ())
    positions = _positions(farm)
    while len(out["hands"]) < len(positions) - 1:
        out["hands"].append(["PASS"])
    invs = _inventories(observation, len(positions))
    cp, _, ca, _ = _asset_targets(farm)
    changes = []
    water = feed = 0
    for unit_idx, pos in enumerate(positions):
        planned = out["farmer"] if unit_idx == 0 else out["hands"][unit_idx - 1]
        if str(planned[0]) != "PASS":
            continue
        tile = _tile_at(farm, pos)
        inv = invs[unit_idx] if unit_idx < len(invs) else {}
        replacement = None
        if (
            isinstance(tile, dict)
            and tile.get("kind") == "PLANT"
            and not tile.get("watered_today", False)
            and int(tile.get("consecutive_unwatered", 0) or 0) >= 1
        ):
            replacement = ["WATER"]
            water += 1
        elif (
            isinstance(tile, dict)
            and "animal" in tile
            and not tile.get("fed_today", False)
            and int(tile.get("consecutive_unfed", 0) or 0) >= 1
            and int(_get(inv, "WHEAT", 0) or 0) > 0
        ):
            replacement = ["FEED"]
            feed += 1
        if replacement is not None:
            before = _set_unit_action(out, unit_idx, replacement)
            changes.append((unit_idx, before, tuple(replacement)))
    return out, FarmCareResult(
        applied=bool(changes),
        critical_plants=len(cp),
        critical_animals=len(ca),
        overrides=len(changes),
        water_orders=water,
        feed_orders=feed,
        care_orders=0,
        fertilize_orders=0,
        wheat_rescue_buys=0,
        unit_changes=tuple(changes),
    )


def _idle_for_rest_of_route(route, hand_idx, step):
    """True only when the static route never uses this hand again.

    This is intentionally stricter than 'idle until day end'. A hand moved by
    the learned micro-policy must never be expected at a fixed macro position
    on a later day, otherwise one useful care action can desynchronize the
    entire remaining choreography.
    """
    if not isinstance(route, (list, tuple)):
        return False
    for future in range(int(step), len(route)):
        raw = route[future] if 0 <= future < len(route) else {}
        hands = (raw or {}).get("hands") or []
        planned = hands[hand_idx] if hand_idx < len(hands) else ["PASS"]
        if str(_normalize_action(planned)[0]) != "PASS":
            return False
    return True


def safe_idle_hand_indices(route, step, hand_count, turns_per_day=24):
    del turns_per_day  # kept for call-site compatibility
    return tuple(
        idx
        for idx in range(max(0, int(hand_count)))
        if _idle_for_rest_of_route(route, idx, step)
    )


def _all_targets(farm, day):
    targets = {
        "water": [], "feed": [], "care": [], "fert_source": [],
        "fertilize": [], "harvest": [], "weed": [],
    }
    for y, row in enumerate(_get(farm, "tiles", []) or []):
        for x, tile in enumerate(row or []):
            pos = (x, y)
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                targets["weed"].append(pos)
                continue
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                if not tile.get("watered_today", False):
                    priority = int(tile.get("consecutive_unwatered", 0) or 0)
                    targets["water"].append((priority, pos))
                if int(tile.get("fertilized_until_day", -1) or -1) < int(day):
                    targets["fertilize"].append(pos)
                if _ready_harvest(tile, day):
                    targets["harvest"].append(pos)
            if isinstance(tile, dict) and "animal" in tile:
                if not tile.get("fed_today", False):
                    priority = int(tile.get("consecutive_unfed", 0) or 0)
                    targets["feed"].append((priority, pos))
                if tile.get("fed_today", False) and not tile.get("cared_today", False):
                    targets["care"].append(pos)
                if tile.get("fertilizer_available", False):
                    targets["fert_source"].append(pos)
                if _ready_harvest(tile, day):
                    targets["harvest"].append(pos)
    targets["water"] = [p for _, p in sorted(targets["water"], key=lambda x: (-x[0], x[1][1], x[1][0]))]
    targets["feed"] = [p for _, p in sorted(targets["feed"], key=lambda x: (-x[0], x[1][1], x[1][0]))]
    return targets


def micro_local_features(observation, hand_idx):
    farm, _ = _farm(observation)
    if farm is None:
        return [0.0] * MICRO_LOCAL_DIM
    positions = _positions(farm)
    unit_idx = int(hand_idx) + 1
    if not (0 <= unit_idx < len(positions)):
        return [0.0] * MICRO_LOCAL_DIM
    pos = positions[unit_idx]
    tile = _tile_at(farm, pos)
    invs = _inventories(observation, len(positions))
    inv = invs[unit_idx] if unit_idx < len(invs) else {}
    shed = _shed(observation)
    day = int(_get(observation, "day", 0) or 0)
    hour = int(_get(observation, "hour", 0) or 0)
    targets = _all_targets(farm, day)
    board = max(1, len(_get(farm, "tiles", []) or []))
    access = _shed_access(board)
    def dmin(rows):
        return min((_distance(pos, p) for p in rows), default=2 * board) / max(1.0, 2.0 * board)
    kind_plant = float(isinstance(tile, dict) and tile.get("kind") == "PLANT")
    kind_animal = float(isinstance(tile, dict) and "animal" in tile)
    kind_weed = float(isinstance(tile, dict) and tile.get("kind") == "WEED")
    tiles = _get(farm, "tiles", []) or []
    empty_tiles = []
    coop_slots = []
    pasture_slots = []
    for y, row in enumerate(tiles):
        for x, cell in enumerate(row or []):
            if cell is None:
                empty_tiles.append((x, y))
            elif isinstance(cell, dict):
                if cell.get("kind") == "COOP" and "animal" not in cell:
                    coop_slots.append((x, y))
                elif cell.get("kind") == "PASTURE" and "animal" not in cell:
                    pasture_slots.append((x, y))
    seeds = _get(_private(observation), "seeds", {}) or {}
    seed_total = sum(max(0, int(v or 0)) for v in seeds.values())
    pending_animals = {
        animal: int(_get(inv, animal, 0) or 0) + int(_get(shed, animal, 0) or 0)
        for animal in ("GOOSE", "COW", "SHEEP")
    }
    market = _get(observation, "market", {}) or {}
    prices = _get(market, "prices", {}) or {}
    sale_products = (
        "WHEAT", "CARROT", "TOMATO", "STRAWBERRY",
        "MELON", "EGG", "MILK", "WOOL",
    )
    carried_sale_value = sum(
        max(0, int(_get(inv, p, 0) or 0))
        * max(0.0, float(_get(prices, p, 0.0) or 0.0))
        for p in sale_products
    )
    tile_yield = float(tile.get("yield_units", 0) or 0) if isinstance(tile, dict) else 0.0
    consecutive_unwatered = float(tile.get("consecutive_unwatered", 0) or 0) if isinstance(tile, dict) else 0.0
    consecutive_unfed = float(tile.get("consecutive_unfed", 0) or 0) if isinstance(tile, dict) else 0.0
    fertilizer_available = float(bool(isinstance(tile, dict) and tile.get("fertilizer_available", False)))
    money = max(0.0, float(_get(farm, "money", 0.0) or 0.0))
    features = [
        float(pos[0]) / max(1, board - 1),
        float(pos[1]) / max(1, board - 1),
        kind_plant,
        kind_animal,
        kind_weed,
        float(tile is None),
        float(bool(isinstance(tile, dict) and tile.get("watered_today", False))),
        float(bool(isinstance(tile, dict) and tile.get("fed_today", False))),
        float(bool(isinstance(tile, dict) and tile.get("cared_today", False))),
        min(1.0, float(_get(inv, "WHEAT", 0) or 0) / 10.0),
        min(1.0, float(_get(inv, "FERTILIZER", 0) or 0) / 10.0),
        min(1.0, float(_get(shed, "WHEAT", 0) or 0) / 50.0),
        min(1.0, float(_get(shed, "FERTILIZER", 0) or 0) / 50.0),
        min(1.0, len(targets["water"]) / 25.0),
        min(1.0, len(targets["feed"]) / 25.0),
        min(1.0, len(targets["care"]) / 25.0),
        min(1.0, len(targets["harvest"]) / 25.0),
        min(1.0, len(targets["weed"]) / 25.0),
        dmin(targets["water"]),
        dmin(targets["feed"]),
        min((_distance(pos, p) for p in access), default=2 * board) / max(1.0, 2.0 * board),
        min(1.0, len(targets["fert_source"]) / 25.0),
        min(1.0, sum(
            int(_get(inv, p, 0) or 0)
            for p in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL")
        ) / 20.0),
        float(hour) / 23.0,
        float(day) / 29.0,
        dmin(targets["care"]),
        dmin(targets["harvest"]),
        dmin(targets["fert_source"]),
        dmin(targets["fertilize"]),
        dmin(targets["weed"]),
        dmin(empty_tiles),
        dmin(coop_slots),
        dmin(pasture_slots),
        min(1.0, float(seed_total) / 20.0),
        min(1.0, float(pending_animals["GOOSE"]) / 4.0),
        min(1.0, float(pending_animals["COW"]) / 4.0),
        min(1.0, float(pending_animals["SHEEP"]) / 4.0),
        min(1.0, max(0.0, tile_yield) / 20.0),
        min(1.0, max(0.0, consecutive_unwatered) / 2.0),
        min(1.0, max(0.0, consecutive_unfed) / 2.0),
        fertilizer_available,
        min(1.0, carried_sale_value / 20000.0),
        min(1.0, money / 100000.0),
    ]
    if len(features) != MICRO_LOCAL_DIM:
        raise RuntimeError(f"micro local width mismatch: {len(features)}")
    return features


def micro_plant_context_features(observation, hand_idx):
    """Plant lifecycle context kept out of the legacy 43D local schema.

    This branch is zero-residual at migration, so old checkpoints retain their
    exact behavior until BC learns to use crop identity and growth timing.
    """
    out = [0.0] * PLANT_CONTEXT_DIM
    farm, _ = _farm(observation)
    if farm is None:
        return out
    positions = _positions(farm)
    unit_idx = int(hand_idx) + 1
    if not (0 <= unit_idx < len(positions)):
        return out
    tile = _tile_at(farm, positions[unit_idx])
    if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
        return out
    crop = str(tile.get("crop", ""))
    crops = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
    if crop not in CROP_META:
        return out
    out[crops.index(crop)] = 1.0
    meta = CROP_META[crop]
    day = int(_get(observation, "day", 0) or 0)
    planted = int(tile.get("planted_day", day) or day)
    age = max(0.0, float(day - planted))
    first = max(1.0, float(meta["first"]))
    interval = max(1.0, float(meta.get("interval", 1.0) or 1.0))
    max_yield = max(1.0, float(meta["max_yield"]))
    ongoing = bool(meta.get("ongoing", False))
    if age < first:
        to_next = first - age
    elif ongoing:
        phase = (age - first) % interval
        to_next = 0.0 if phase == 0 else interval - phase
    else:
        to_next = 0.0
    out[5] = min(1.0, age / 12.0)
    out[6] = min(1.0, age / first)
    out[7] = float(_ready_harvest(tile, day))
    out[8] = float(ongoing)
    out[9] = min(1.0, max(0.0, float(tile.get("yield_units", 0) or 0)) / max_yield)
    out[10] = float(int(tile.get("fertilized_until_day", -1) or -1) >= day)
    out[11] = min(1.0, max(0.0, to_next) / 10.0)
    return out


def _nearest_target(pos, rows):
    return min(rows, key=lambda p: (_distance(pos, p), p[1], p[0])) if rows else None

def _strategy_scores(observation):
    market = _get(observation, "market", {}) or {}
    prices = _get(market, "prices", {}) or {}
    town = _get(observation, "town", {}) or {}
    shops = list(_get(town, "unlocked_shops", []) or [])
    farms = list(_get(observation, "farms", []) or [])
    player = int(_get(observation, "player", 0) or 0)
    rival = farms[1 - player] if len(farms) >= 2 and 0 <= player < 2 else {}
    shop_products = {
        "BAKERY": ("EGG", "WHEAT"),
        "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
        "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
        "YARN_STORE": ("WOOL",),
        "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
        "PET_CAFE": ("CARROT",),
        "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
        "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
    }
    base = {
        "WHEAT": 25.0, "CARROT": 35.0, "TOMATO": 60.0,
        "STRAWBERRY": 120.0, "MELON": 250.0, "EGG": 50.0,
        "MILK": 160.0, "WOOL": 200.0,
    }
    demand = {p: 0.0 for p in base}
    for shop in shops:
        products = shop_products.get(str(shop), ())
        mult = 2.0 if len(products) == 1 else 1.0
        for product in products:
            if product in demand:
                demand[product] += mult
    pressure = {p: 0.0 for p in base}
    animal_product = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
    for row in _get(rival, "tiles", []) or []:
        for tile in row or []:
            if not isinstance(tile, dict):
                continue
            crop = tile.get("crop")
            animal = tile.get("animal")
            y = max(0.0, float(tile.get("yield_units", 0) or 0))
            if crop in pressure:
                pressure[crop] += 1.0 + y
            product = animal_product.get(animal)
            if product in pressure:
                pressure[product] += 1.5 + y
    crop_scores = {}
    for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"):
        ratio = float(_get(prices, crop, base[crop]) or 0.0) / max(1.0, base[crop])
        crop_scores[crop] = ratio + 0.28 * demand[crop] - 0.035 * pressure[crop]
    animal_scores = {}
    for animal, product in animal_product.items():
        ratio = float(_get(prices, product, base[product]) or 0.0) / max(1.0, base[product])
        animal_scores[animal] = ratio + 0.34 * demand[product] - 0.035 * pressure[product]
    return crop_scores, animal_scores, demand, pressure


def _remaining_days(observation):
    step = int(_get(observation, "step", 0) or 0)
    return max(0.0, (719 - step) / 24.0)


def _best_seed_crop(observation):
    seeds = _get(_private(observation), "seeds", {}) or {}
    crop_scores, _, _, _ = _strategy_scores(observation)
    first_yield = {
        "WHEAT": 2, "CARROT": 2, "TOMATO": 8,
        "STRAWBERRY": 10, "MELON": 10,
    }
    days = _remaining_days(observation)
    candidates = [
        crop for crop, qty in seeds.items()
        if int(qty or 0) > 0 and crop in crop_scores and days >= first_yield[crop] + 0.5
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda crop: (crop_scores[crop], crop))


def _best_pending_animal(observation, inv):
    shed = _shed(observation)
    _, animal_scores, _, _ = _strategy_scores(observation)
    first_yield = {"GOOSE": 4, "COW": 8, "SHEEP": 6}
    days = _remaining_days(observation)
    candidates = []
    for animal in ("GOOSE", "COW", "SHEEP"):
        carried = int(_get(inv, animal, 0) or 0)
        stored = int(_get(shed, animal, 0) or 0)
        if carried + stored > 0 and days >= first_yield[animal] + 0.5:
            candidates.append(animal)
    if not candidates:
        return None
    return max(candidates, key=lambda animal: (animal_scores[animal], animal))


def compile_micro_task(observation, hand_idx, task):
    """Compile a learned task into one safe unit action for an idle hand."""
    farm, _ = _farm(observation)
    if farm is None:
        return ["PASS"], {"valid": False, "distance": 0}
    positions = _positions(farm)
    unit_idx = int(hand_idx) + 1
    if not (0 <= unit_idx < len(positions)):
        return ["PASS"], {"valid": False, "distance": 0}
    pos = positions[unit_idx]
    tile = _tile_at(farm, pos)
    invs = _inventories(observation, len(positions))
    inv = invs[unit_idx] if unit_idx < len(invs) else {}
    shed = _shed(observation)
    day = int(_get(observation, "day", 0) or 0)
    board = max(1, len(_get(farm, "tiles", []) or []))
    targets = _all_targets(farm, day)
    task = str(task)
    if task == "KEEP":
        return ["PASS"], {"valid": True, "distance": 0}

    target = None
    if task == "WATER":
        if isinstance(tile, dict) and tile.get("kind") == "PLANT" and not tile.get("watered_today", False):
            return ["WATER"], {"valid": True, "distance": 0}
        critical = [
            p for p in targets["water"]
            if isinstance(_tile_at(farm, p), dict)
            and int(_tile_at(farm, p).get("consecutive_unwatered", 0) or 0) >= 1
        ]
        target = _nearest_target(pos, critical or targets["water"])
    elif task == "FEED":
        wheat = int(_get(inv, "WHEAT", 0) or 0)
        if isinstance(tile, dict) and "animal" in tile and not tile.get("fed_today", False) and wheat > 0:
            return ["FEED"], {"valid": True, "distance": 0}
        if wheat <= 0:
            access = min(_shed_access(board), key=lambda p: _distance(pos, p))
            if tuple(pos) == tuple(access) and int(_get(shed, "WHEAT", 0) or 0) > 0:
                return ["PICKUP", "WHEAT", min(8, int(_get(shed, "WHEAT", 0) or 0))], {"valid": True, "distance": 0}
            target = access
        else:
            critical = [
                p for p in targets["feed"]
                if isinstance(_tile_at(farm, p), dict)
                and int(_tile_at(farm, p).get("consecutive_unfed", 0) or 0) >= 1
            ]
            target = _nearest_target(pos, critical or targets["feed"])
    elif task == "CARE":
        if isinstance(tile, dict) and "animal" in tile and tile.get("fed_today", False) and not tile.get("cared_today", False):
            return ["CARE"], {"valid": True, "distance": 0}
        target = _nearest_target(pos, targets["care"])
    elif task == "COLLECT_FERT":
        if isinstance(tile, dict) and "animal" in tile and tile.get("fertilizer_available", False):
            return ["COLLECT_FERTILIZER"], {"valid": True, "distance": 0}
        target = _nearest_target(pos, targets["fert_source"])
    elif task == "FERTILIZE":
        fert = int(_get(inv, "FERTILIZER", 0) or 0)
        if fert > 0 and isinstance(tile, dict) and tile.get("kind") == "PLANT" and int(tile.get("fertilized_until_day", -1) or -1) < day:
            return ["FERTILIZE"], {"valid": True, "distance": 0}
        target = _nearest_target(pos, targets["fertilize"] if fert > 0 else targets["fert_source"])
    elif task == "HARVEST":
        if _ready_harvest(tile, day):
            return ["HARVEST"], {"valid": True, "distance": 0}
        target = _nearest_target(pos, targets["harvest"])
    elif task == "STORE_SHED":
        access = min(_shed_access(board), key=lambda p: _distance(pos, p))
        market = _get(observation, "market", {}) or {}
        prices = _get(market, "prices", {}) or {}
        sale_items = [
            p for p in ("CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL")
            if int(_get(inv, p, 0) or 0) > 0
        ]
        wheat = int(_get(inv, "WHEAT", 0) or 0)
        # Preserve a small feed reserve on the hand; only excess wheat is a
        # storage candidate.
        if wheat > 8:
            sale_items.append("WHEAT")
        if not sale_items:
            return ["PASS"], {"valid": False, "distance": 0}
        if tuple(pos) == tuple(access):
            item = max(
                sale_items,
                key=lambda p: (
                    float(_get(prices, p, 0.0) or 0.0),
                    int(_get(inv, p, 0) or 0),
                    p,
                ),
            )
            qty = int(_get(inv, item, 0) or 0)
            if item == "WHEAT":
                qty = max(0, qty - 8)
            if qty > 0:
                return ["PLACE", item, qty], {"valid": True, "distance": 0}
            return ["PASS"], {"valid": False, "distance": 0}
        target = access
    elif task == "FETCH_WHEAT":
        access = min(_shed_access(board), key=lambda p: _distance(pos, p))
        if tuple(pos) == tuple(access) and int(_get(shed, "WHEAT", 0) or 0) > 0:
            return ["PICKUP", "WHEAT", min(8, int(_get(shed, "WHEAT", 0) or 0))], {"valid": True, "distance": 0}
        target = access
    elif task == "DIG_WEED":
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            return ["DIG"], {"valid": True, "distance": 0}
        target = _nearest_target(pos, targets["weed"])
    elif task == "PLANT_BEST":
        crop = _best_seed_crop(observation)
        if crop is None:
            return ["PASS"], {"valid": False, "distance": 0}
        if tile is None:
            return ["PLANT", crop], {"valid": True, "distance": 0, "crop": crop}
        empty = [
            (x, y)
            for y, row in enumerate(_get(farm, "tiles", []) or [])
            for x, cell in enumerate(row or [])
            if cell is None
        ]
        target = _nearest_target(pos, empty)
        if target is None:
            return ["PASS"], {"valid": False, "distance": 0}
    elif task == "DEPLOY_ANIMAL":
        animal = _best_pending_animal(observation, inv)
        if animal is None:
            return ["PASS"], {"valid": False, "distance": 0}
        structure = "COOP" if animal == "GOOSE" else "PASTURE"
        carrying = int(_get(inv, animal, 0) or 0) > 0
        if not carrying:
            access = min(_shed_access(board), key=lambda p: _distance(pos, p))
            if tuple(pos) == tuple(access) and int(_get(shed, animal, 0) or 0) > 0:
                return ["PICKUP", animal, 1], {
                    "valid": True, "distance": 0, "animal": animal,
                }
            target = access
        else:
            if (
                isinstance(tile, dict)
                and tile.get("kind") == structure
                and "animal" not in tile
            ):
                return ["PLACE", animal, 1], {
                    "valid": True, "distance": 0, "animal": animal,
                }
            structures = [
                (x, y)
                for y, row in enumerate(_get(farm, "tiles", []) or [])
                for x, cell in enumerate(row or [])
                if isinstance(cell, dict)
                and cell.get("kind") == structure
                and "animal" not in cell
            ]
            if structures:
                target = _nearest_target(pos, structures)
            elif tile is None:
                return ["BUILD_COOP" if structure == "COOP" else "BUILD_PASTURE"], {
                    "valid": True, "distance": 0, "animal": animal,
                }
            else:
                empty = [
                    (x, y)
                    for y, row in enumerate(_get(farm, "tiles", []) or [])
                    for x, cell in enumerate(row or [])
                    if cell is None
                ]
                target = _nearest_target(pos, empty)
                if target is None:
                    return ["PASS"], {"valid": False, "distance": 0}
    if target is None:
        return ["PASS"], {"valid": False, "distance": 0}
    return _step_toward(pos, target), {
        "valid": True,
        "distance": int(_distance(pos, target)),
        "target": tuple(target),
    }


def compile_v46_skill_task(observation, hand_idx, task):
    """Compile V4.6 full-skill semantics without changing legacy micro behavior.

    COLLECT_FERT owns every fertilizer acquisition path (animal collection and
    shed pickup). FERTILIZE only means applying fertilizer while already
    carrying it. This removes the old semantic collision where FERTILIZE also
    meant navigating to fertilizer sources.
    """
    task = str(task)
    if task not in {"COLLECT_FERT", "FERTILIZE"}:
        return compile_micro_task(observation, hand_idx, task)

    farm, _ = _farm(observation)
    if farm is None:
        return ["PASS"], {"valid": False, "distance": 0}
    positions = _positions(farm)
    unit_idx = int(hand_idx) + 1
    if not (0 <= unit_idx < len(positions)):
        return ["PASS"], {"valid": False, "distance": 0}
    pos = positions[unit_idx]
    tile = _tile_at(farm, pos)
    invs = _inventories(observation, len(positions))
    inv = invs[unit_idx] if unit_idx < len(invs) else {}
    shed = _shed(observation)
    day = int(_get(observation, "day", 0) or 0)
    board = max(1, len(_get(farm, "tiles", []) or []))
    targets = _all_targets(farm, day)

    if task == "COLLECT_FERT":
        if (
            isinstance(tile, dict)
            and "animal" in tile
            and bool(tile.get("fertilizer_available", False))
        ):
            return ["COLLECT_FERTILIZER"], {"valid": True, "distance": 0}

        sources = list(targets.get("fert_source") or [])
        shed_qty = int(_get(shed, "FERTILIZER", 0) or 0)
        access = min(_shed_access(board), key=lambda p: _distance(pos, p))
        source = _nearest_target(pos, sources)
        source_dist = _distance(pos, source) if source is not None else 10**9
        shed_dist = _distance(pos, access) if shed_qty > 0 else 10**9

        if shed_qty > 0 and shed_dist <= source_dist:
            if tuple(pos) == tuple(access):
                return ["PICKUP", "FERTILIZER", min(8, shed_qty)], {
                    "valid": True, "distance": 0, "source": "shed"
                }
            target = access
        elif source is not None:
            target = source
        else:
            return ["PASS"], {"valid": False, "distance": 0}

        return _step_toward(pos, target), {
            "valid": True,
            "distance": int(_distance(pos, target)),
            "target": tuple(target),
        }

    # FERTILIZE is application-only in V4.6. Acquisition is COLLECT_FERT.
    fert = int(_get(inv, "FERTILIZER", 0) or 0)
    if fert <= 0:
        return ["PASS"], {"valid": False, "distance": 0}
    if (
        isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and int(tile.get("fertilized_until_day", -1) or -1) < day
    ):
        return ["FERTILIZE"], {"valid": True, "distance": 0}
    target = _nearest_target(pos, targets.get("fertilize") or [])
    if target is None:
        return ["PASS"], {"valid": False, "distance": 0}
    return _step_toward(pos, target), {
        "valid": True,
        "distance": int(_distance(pos, target)),
        "target": tuple(target),
    }


def _ready_harvest(tile, day):
    if not isinstance(tile, dict) or int(tile.get("yield_units", 0) or 0) <= 0:
        return False
    if "animal" in tile:
        return True
    if tile.get("kind") != "PLANT":
        return False
    crop = str(tile.get("crop", ""))
    first = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}.get(crop, 0)
    return int(day) - int(tile.get("planted_day", 0) or 0) >= first


def optimize_safe_idle_farm_labor(
    observation,
    action,
    *,
    route,
    step,
    turns_per_day=24,
):
    """Use only hands whose remaining route tape for this day is all PASS."""
    out = {
        "farmer": _normalize_action((action or {}).get("farmer")),
        "hands": [_normalize_action(x) for x in ((action or {}).get("hands") or [])],
        "market": [list(x) for x in ((action or {}).get("market") or []) if isinstance(x, (list, tuple))],
    }
    farm, _ = _farm(observation)
    if farm is None:
        return out, IdleLaborResult(False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    positions = _positions(farm)
    while len(out["hands"]) < len(positions) - 1:
        out["hands"].append(["PASS"])
    idle = list(
        safe_idle_hand_indices(
            route, step, len(positions) - 1, turns_per_day=turns_per_day
        )
    )
    if not idle:
        return out, IdleLaborResult(False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    invs = _inventories(observation, len(positions))
    shed = _shed(observation)
    day = int(_get(observation, "day", 0) or 0)
    tiles = _get(farm, "tiles", []) or []
    market = _get(observation, "market", {}) or {}
    prices = _get(market, "prices", {}) or {}
    cp, dp, ca, da = _asset_targets(farm)
    harvest = []
    care_targets = []
    fert_sources = []
    fert_targets = []
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row or []):
            if _ready_harvest(tile, day):
                harvest.append((x, y))
            if isinstance(tile, dict) and "animal" in tile:
                if tile.get("fed_today", False) and not tile.get("cared_today", False):
                    care_targets.append((x, y))
                if tile.get("fertilizer_available", False):
                    fert_sources.append((x, y))
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                if int(tile.get("fertilized_until_day", -1) or -1) < day:
                    fert_targets.append((x, y))

    claimed = set()
    moved = watered = fed = cared = harvested = collected = fertilized = pickups = 0
    access = _shed_access(len(tiles))
    for hand_idx in idle:
        unit_idx = hand_idx + 1
        pos = positions[unit_idx]
        inv = invs[unit_idx] if unit_idx < len(invs) else {}
        tile = _tile_at(farm, pos)
        replacement = None

        if isinstance(tile, dict) and "animal" in tile:
            critical = int(tile.get("consecutive_unfed", 0) or 0) >= 1
            if not tile.get("fed_today", False) and int(_get(inv, "WHEAT", 0) or 0) > 0:
                replacement = ["FEED"]; fed += 1
            elif tile.get("fed_today", False) and not tile.get("cared_today", False):
                replacement = ["CARE"]; cared += 1
            elif tile.get("fertilizer_available", False):
                replacement = ["COLLECT_FERTILIZER"]; collected += 1
            elif critical:
                replacement = ["PASS"]
        elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
            critical = int(tile.get("consecutive_unwatered", 0) or 0) >= 1
            if not tile.get("watered_today", False):
                replacement = ["WATER"]; watered += 1
            elif _ready_harvest(tile, day):
                replacement = ["HARVEST"]; harvested += 1
            elif int(_get(inv, "FERTILIZER", 0) or 0) > 0 and int(tile.get("fertilized_until_day", -1) or -1) < day:
                replacement = ["FERTILIZE"]; fertilized += 1

        if replacement is None:
            has_wheat = int(_get(inv, "WHEAT", 0) or 0) > 0
            has_fert = int(_get(inv, "FERTILIZER", 0) or 0) > 0
            target = None
            if has_wheat:
                target = _nearest(pos, ca or da, claimed)
            if target is None:
                target = _nearest(pos, cp or dp, claimed)
            if target is None:
                target = _nearest(pos, harvest, claimed)
            if target is None and has_fert:
                scored = sorted(
                    [t for t in fert_targets if t not in claimed],
                    key=lambda t: -float(_get(prices, str(_tile_at(farm, t).get("crop", "")), 0) or 0),
                )
                target = scored[0] if scored else None
            if target is None:
                target = _nearest(pos, care_targets, claimed)
            if target is None and not has_fert:
                target = _nearest(pos, fert_sources, claimed)

            need_wheat = bool((ca or da) and not has_wheat)
            if target is None and need_wheat:
                nearest_access = min(access, key=lambda p: _distance(pos, p))
                if tuple(pos) == tuple(nearest_access) and int(_get(shed, "WHEAT", 0) or 0) > 0:
                    replacement = ["PICKUP", "WHEAT", min(8, int(_get(shed, "WHEAT", 0) or 0))]
                    pickups += 1
                else:
                    target = nearest_access

            if replacement is None and target is not None:
                if tuple(pos) == tuple(target):
                    replacement = ["PASS"]
                else:
                    replacement = _step_toward(pos, target)
                    moved += 1
                    claimed.add(tuple(target))

        if replacement is not None:
            out["hands"][hand_idx] = replacement

    carry_wheat = sum(int(_get(invs[i + 1], "WHEAT", 0) or 0) for i in idle if i + 1 < len(invs))
    rescue_need = max(0, len(ca) - carry_wheat - int(_get(shed, "WHEAT", 0) or 0))
    rescue_buys = 0
    if rescue_need > 0 and len(out["market"]) < 10:
        rescue_buys = min(8, rescue_need)
        out["market"].append(["BUY_PRODUCT", "WHEAT", rescue_buys])

    return out, IdleLaborResult(
        applied=bool(moved or watered or fed or cared or harvested or collected or fertilized or pickups or rescue_buys),
        idle_hands=len(idle),
        moved=moved,
        watered=watered,
        fed=fed,
        cared=cared,
        harvested=harvested,
        collected_fertilizer=collected,
        fertilized=fertilized,
        pickups=pickups,
        rescue_buys=rescue_buys,
    )


def apply_feasible_investment_overlay(
    observation,
    action,
    *,
    crop_improvement_ratio=1.20,
    min_crop_roi=1.05,
    allow_crop_switch=False,
    filter_animal_buys=True,
):
    """Conservatively rewrite only existing investment actions.

    No new unit/market slot is created. BUY_SEED can switch to a materially
    better harvest-feasible crop; PLANT can use only seed already owned before
    the current step; unharvestable investments are dropped. Animal purchases
    are filtered only when care/time feasibility fails, preserving structures.
    """
    farm, _ = _farm(observation)
    if farm is None:
        return action, {"applied": False, "error": "no_farm"}
    snap = strategy_snapshot(observation)
    private = _private(observation)
    seeds = {crop: int(_get(_get(private, "seeds", {}) or {}, crop, 0) or 0) for crop in CROP_META}
    shed = _shed(observation)
    inventories = _inventories(observation, 1 + len(_get(farm, "hands", []) or []))

    active_animals = sum(
        1
        for row in (_get(farm, "tiles", []) or [])
        for tile in (row or [])
        if isinstance(tile, dict) and tile.get("animal") in ANIMAL_META
    )
    pending_animals = sum(
        int(_get(shed, animal, 0) or 0)
        + sum(int(_get(inv, animal, 0) or 0) for inv in inventories)
        for animal in ANIMAL_META
    )
    protect_wheat = (active_animals + pending_animals) > 0

    out = {
        "farmer": _normalize_action((action or {}).get("farmer")),
        "hands": [_normalize_action(x) for x in ((action or {}).get("hands") or [])],
        "market": [list(x) for x in ((action or {}).get("market") or []) if isinstance(x, (list, tuple)) and x],
    }
    meta = {
        "applied": False, "plant_switches": 0, "plant_drops": 0,
        "seed_switches": 0, "seed_drops": 0, "animal_buy_drops": 0,
        "events": [],
    }

    viable_crops = [
        crop for crop in CROP_META
        if bool(snap["crop_viable"].get(crop, False))
        and float(snap["crop_roi"].get(crop, 0.0)) >= float(min_crop_roi)
    ]

    def best_crop_for(original, *, require_owned):
        original = str(original)
        orig_roi = float(snap["crop_roi"].get(original, 0.0))
        orig_viable = (
            bool(snap["crop_viable"].get(original, False))
            and orig_roi >= float(min_crop_roi)
            and (not require_owned or seeds.get(original, 0) > 0)
        )
        # Default V4.6 behavior is filter-only. The macro route keeps its crop
        # identity; economics may veto an investment that cannot pay back, but
        # must not silently change the farm composition/choreography.
        if orig_viable:
            return original
        if not bool(allow_crop_switch):
            return None
        if protect_wheat and original == "WHEAT" and bool(snap["crop_viable"].get("WHEAT", False)):
            return "WHEAT"
        candidates = [c for c in viable_crops if (not require_owned or seeds.get(c, 0) > 0)]
        if not candidates:
            return None
        best = max(
            candidates,
            key=lambda c: (float(snap["crop_roi"][c]), -float(CROP_META[c]["first"]), c),
        )
        if orig_roi > 0.0 and float(snap["crop_roi"][best]) < orig_roi * float(crop_improvement_ratio):
            return None
        return best

    # Market executes after unit actions in the engine. Seed substitutions here
    # are therefore for future steps only, never used by same-step PLANT.
    filtered_market = []
    for order in out["market"]:
        op = str(order[0]) if order else ""
        if op == "BUY_SEED" and len(order) >= 3:
            original = str(order[1])
            replacement = best_crop_for(original, require_owned=False)
            if replacement is None:
                meta["seed_drops"] += 1
                meta["applied"] = True
                meta["events"].append({
                    "kind": "drop_seed", "crop": original,
                    "roi": float(snap["crop_roi"].get(original, 0.0)),
                    "remaining_days": float(snap["remaining_days"]),
                })
                continue
            if replacement != original:
                order = list(order)
                order[1] = replacement
                meta["seed_switches"] += 1
                meta["applied"] = True
            filtered_market.append(order)
            continue
        if op == "BUY_ANIMAL" and len(order) >= 3 and bool(filter_animal_buys):
            animal = str(order[1])
            if (
                animal in ANIMAL_META
                and (
                    not bool(snap["animal_viable"].get(animal, False))
                    or float(snap["care_headroom"]) < 1.60
                )
            ):
                meta["animal_buy_drops"] += 1
                meta["applied"] = True
                meta["events"].append({
                    "kind": "drop_animal", "animal": animal,
                    "roi": float(snap["animal_roi"].get(animal, 0.0)),
                    "care_headroom": float(snap["care_headroom"]),
                })
                continue
        filtered_market.append(order)
    out["market"] = filtered_market

    # Allocate existing seed atomically across current PLANT requests. The engine
    # blocks *all* requests for a crop when demand exceeds stock, so each kept or
    # switched request decrements this virtual stock before the next unit.
    available = dict(seeds)
    units = [("farmer", -1, out["farmer"])] + [("hand", i, x) for i, x in enumerate(out["hands"])]
    for kind, idx, unit_action in units:
        if not unit_action or str(unit_action[0]) != "PLANT" or len(unit_action) < 2:
            continue
        original = str(unit_action[1])
        # best_crop_for reads `seeds`; temporarily expose remaining inventory.
        original_seeds = seeds
        seeds = available
        replacement = best_crop_for(original, require_owned=True)
        seeds = original_seeds
        if replacement is None or available.get(replacement, 0) <= 0:
            replacement = original if (
                available.get(original, 0) > 0
                and bool(snap["crop_viable"].get(original, False))
            ) else None
        if replacement is None:
            new_action = ["PASS"]
            meta["plant_drops"] += 1
            meta["applied"] = True
            meta["events"].append({
                "kind": "drop_plant", "crop": original, "unit": int(idx),
                "roi": float(snap["crop_roi"].get(original, 0.0)),
                "remaining_days": float(snap["remaining_days"]),
            })
        else:
            new_action = ["PLANT", replacement]
            available[replacement] = max(0, available.get(replacement, 0) - 1)
            if replacement != original:
                meta["plant_switches"] += 1
                meta["applied"] = True
        if kind == "farmer":
            out["farmer"] = new_action
        else:
            out["hands"][idx] = new_action

    meta["best_crop"] = (
        max(viable_crops, key=lambda c: float(snap["crop_roi"][c]))
        if viable_crops else None
    )
    meta["care_headroom"] = float(snap["care_headroom"])
    return out, meta


def apply_portfolio_switch_overlay(
    observation,
    action,
    *,
    crop_improvement_ratio=1.30,
    feed_reserve_days=2.0,
    activation_step=144,
    projected_horizon_extra_days=2.0,
    min_undersupply_ratio=0.0,
    min_shop_demand=1.0,
    source_mode="any",
):
    """Rewrite existing seed/plant targets using forward market economics.

    This overlay is intentionally switch-only:
    - no market/unit action slot is added,
    - no action is dropped,
    - no global atomic-plant sanitization is performed,
    - same-step PLANT can switch only to seed already owned,
    - WHEAT is protected whenever near-term feed coverage is insufficient.
    """
    step = int(_get(observation, "step", 0) or 0)
    if step < int(activation_step):
        return action, {
            "applied": False,
            "reason": "before_activation",
            "step": step,
            "events": [],
        }

    town = _get(observation, "town", {}) or {}
    shops = list(_get(town, "unlocked_shops", []) or [])
    if not shops:
        return action, {
            "applied": False,
            "reason": "no_shops",
            "step": step,
            "events": [],
        }

    # Fast path: forward economics is only needed when this frame actually
    # contains an investment target the overlay is allowed to rewrite.
    # This is semantically exact because the overlay never adds action slots.
    source_mode = str(source_mode)
    action_map = action if isinstance(action, dict) else {}
    switchable = False
    for order in list(action_map.get("market") or []):
        if (
            isinstance(order, (list, tuple))
            and len(order) >= 2
            and str(order[0]) == "BUY_SEED"
            and str(order[1]) in CROP_META
            and (
                source_mode != "wheat"
                or str(order[1]) == "WHEAT"
            )
        ):
            switchable = True
            break
    if not switchable:
        commands = [action_map.get("farmer")]
        commands.extend(list(action_map.get("hands") or []))
        for command in commands:
            if (
                isinstance(command, (list, tuple))
                and len(command) >= 2
                and str(command[0]) == "PLANT"
                and str(command[1]) in CROP_META
                and (
                    source_mode != "wheat"
                    or str(command[1]) == "WHEAT"
                )
            ):
                switchable = True
                break
    if not switchable:
        return action, {
            "applied": False,
            "reason": "no_switchable_investment",
            "step": step,
            "events": [],
        }

    farm, _ = _farm(observation)
    if farm is None:
        return action, {"applied": False, "reason": "no_farm", "events": []}

    snap = strategy_snapshot(
        observation,
        projected_crop_value=True,
        projected_crop_horizon_extra_days=float(
            projected_horizon_extra_days
        ),
    )
    private = _private(observation)
    seed_store = _get(private, "seeds", {}) or {}
    seeds = {
        crop: int(_get(seed_store, crop, 0) or 0)
        for crop in CROP_META
    }
    shed = _shed(observation)
    inventories = _inventories(
        observation, 1 + len(_get(farm, "hands", []) or [])
    )

    active_animals = sum(
        1
        for row in (_get(farm, "tiles", []) or [])
        for tile in (row or [])
        if isinstance(tile, dict) and tile.get("animal") in ANIMAL_META
    )
    pending_animals = sum(
        int(_get(shed, animal, 0) or 0)
        + sum(int(_get(inv, animal, 0) or 0) for inv in inventories)
        for animal in ANIMAL_META
    )
    animal_commitment = active_animals + pending_animals
    reserve_days = max(0.0, float(feed_reserve_days))
    feed_need = float(animal_commitment) * reserve_days
    carried_wheat = sum(
        int(_get(inv, "WHEAT", 0) or 0) for inv in inventories
    )
    shed_wheat = int(_get(shed, "WHEAT", 0) or 0)
    near_wheat = float(
        projected_farm_output(
            observation, reserve_days
        ).get("WHEAT", 0.0)
    )
    feed_available = float(carried_wheat + shed_wheat) + near_wheat
    feed_surplus = feed_available - feed_need

    out = {
        "farmer": _normalize_action((action or {}).get("farmer")),
        "hands": [
            _normalize_action(x)
            for x in ((action or {}).get("hands") or [])
        ],
        "market": [
            list(x)
            for x in ((action or {}).get("market") or [])
            if isinstance(x, (list, tuple)) and x
        ],
    }
    meta = {
        "applied": False,
        "step": step,
        "shops": list(shops),
        "seed_switches": 0,
        "plant_switches": 0,
        "feed_need": float(feed_need),
        "feed_available": float(feed_available),
        "feed_surplus": float(feed_surplus),
        "events": [],
    }

    viable = [
        crop
        for crop in CROP_META
        if bool(snap["crop_viable"].get(crop, False))
        and float(snap["demand"].get(crop, 0.0))
        >= float(min_shop_demand)
        and float(snap["crop_undersupply_ratio"].get(crop, 0.0))
        >= float(min_undersupply_ratio)
    ]

    def source_allowed(original):
        mode = str(source_mode)
        if mode == "wheat":
            return original == "WHEAT"
        if mode == "nonbest":
            if not viable:
                return False
            best = max(
                viable,
                key=lambda c: float(snap["crop_roi"].get(c, 0.0)),
            )
            return original != best
        return True

    def choose(original, available_seed=None):
        original = str(original)
        if original not in CROP_META or not source_allowed(original):
            return original
        if (
            original == "WHEAT"
            and animal_commitment > 0
            and feed_surplus <= 0.0
        ):
            return original

        candidates = []
        for crop in viable:
            if crop == original:
                continue
            if available_seed is not None and int(
                available_seed.get(crop, 0)
            ) <= 0:
                continue
            candidates.append(crop)
        if not candidates:
            return original

        best = max(
            candidates,
            key=lambda c: (
                float(snap["crop_roi"].get(c, 0.0)),
                float(snap["crop_projected_price"].get(c, 0.0)),
                -float(CROP_META[c]["first"]),
                c,
            ),
        )
        old_roi = max(
            1e-6, float(snap["crop_roi"].get(original, 0.0))
        )
        new_roi = float(snap["crop_roi"].get(best, 0.0))
        if new_roi < old_roi * float(crop_improvement_ratio):
            return original
        return best

    for index, order in enumerate(list(out["market"])):
        if (
            not order
            or str(order[0]) != "BUY_SEED"
            or len(order) < 3
        ):
            continue
        original = str(order[1])
        replacement = choose(original)
        if replacement == original:
            continue
        new_order = list(order)
        new_order[1] = replacement
        out["market"][index] = new_order
        meta["seed_switches"] += 1
        meta["applied"] = True
        meta["events"].append(
            {
                "kind": "switch_seed",
                "from": original,
                "to": replacement,
                "old_roi": float(snap["crop_roi"].get(original, 0.0)),
                "new_roi": float(snap["crop_roi"].get(replacement, 0.0)),
                "projected_price": float(
                    snap["crop_projected_price"].get(replacement, 0.0)
                ),
                "projected_inventory": float(
                    snap["crop_projected_inventory"].get(
                        replacement, 10000.0
                    )
                ),
                "undersupply_ratio": float(
                    snap["crop_undersupply_ratio"].get(replacement, 0.0)
                ),
            }
        )

    available = dict(seeds)
    units = [("farmer", -1, out["farmer"])] + [
        ("hand", i, x) for i, x in enumerate(out["hands"])
    ]
    for kind, idx, unit_action in units:
        if (
            not unit_action
            or str(unit_action[0]) != "PLANT"
            or len(unit_action) < 2
        ):
            continue
        original = str(unit_action[1])
        replacement = choose(original, available)
        if replacement != original:
            new_action = ["PLANT", replacement]
            available[replacement] = max(
                0, int(available.get(replacement, 0)) - 1
            )
            meta["plant_switches"] += 1
            meta["applied"] = True
            meta["events"].append(
                {
                    "kind": "switch_plant",
                    "from": original,
                    "to": replacement,
                    "unit": int(idx),
                    "old_roi": float(
                        snap["crop_roi"].get(original, 0.0)
                    ),
                    "new_roi": float(
                        snap["crop_roi"].get(replacement, 0.0)
                    ),
                    "projected_price": float(
                        snap["crop_projected_price"].get(
                            replacement, 0.0
                        )
                    ),
                    "undersupply_ratio": float(
                        snap["crop_undersupply_ratio"].get(
                            replacement, 0.0
                        )
                    ),
                }
            )
        else:
            new_action = list(unit_action)
            if original in available and available[original] > 0:
                available[original] -= 1

        if kind == "farmer":
            out["farmer"] = new_action
        else:
            out["hands"][idx] = new_action

    meta["best_crop"] = (
        max(
            viable,
            key=lambda c: float(snap["crop_roi"].get(c, 0.0)),
        )
        if viable
        else None
    )
    return out, meta


def state_aware_market_orders(observation, *, remaining_steps=None):
    """State-aware post-cutover market planner driven by live economics.

    It protects feed/cash reserves, can conservatively buy viable seeds/animals
    or land when projected payback fits the remaining horizon, and sells shed
    inventory using price, shop demand and opponent supply pressure.
    """
    farm, player = _farm(observation)
    if farm is None:
        return []
    farms = list(_get(observation, "farms", []) or [])
    rival = farms[1 - player] if len(farms) >= 2 else {}
    shed = _shed(observation)
    market = _get(observation, "market", {}) or {}
    prices = _get(market, "prices", {}) or {}
    town = _get(observation, "town", {}) or {}
    shops = list(_get(town, "unlocked_shops", []) or [])
    money = float(_get(farm, "money", 0.0) or 0.0)
    remaining = 999 if remaining_steps is None else int(remaining_steps)

    shop_products = {
        "BAKERY": ("EGG", "WHEAT"),
        "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
        "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
        "YARN_STORE": ("WOOL",),
        "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
        "PET_CAFE": ("CARROT",),
        "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
        "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
    }
    base_price = {
        "WHEAT": 25.0, "CARROT": 35.0, "TOMATO": 60.0,
        "STRAWBERRY": 120.0, "MELON": 250.0, "EGG": 50.0,
        "MILK": 160.0, "WOOL": 200.0, "FERTILIZER": 100.0,
    }
    demand = {p: 0.0 for p in base_price}
    for shop in shops:
        products = shop_products.get(str(shop), ())
        mult = 2.0 if len(products) == 1 else 1.0
        for p in products:
            if p in demand:
                demand[p] += mult

    animal_product = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
    animal_count = 0
    for row in _get(farm, "tiles", []) or []:
        for tile in row or []:
            if isinstance(tile, dict) and tile.get("animal") in animal_product:
                animal_count += 1

    rival_pressure = {p: 0.0 for p in base_price}
    for row in _get(rival, "tiles", []) or []:
        for tile in row or []:
            if not isinstance(tile, dict):
                continue
            crop = tile.get("crop")
            animal = tile.get("animal")
            y = max(0.0, float(tile.get("yield_units", 0) or 0))
            if crop in rival_pressure:
                rival_pressure[crop] += 1.0 + y
            product = animal_product.get(animal)
            if product in rival_pressure:
                rival_pressure[product] += 1.5 + y

    inventories = _inventories(observation, 1 + len(_get(farm, "hands", []) or []))
    carried_wheat = sum(int(_get(inv, "WHEAT", 0) or 0) for inv in inventories)
    shed_wheat = int(_get(shed, "WHEAT", 0) or 0)
    reserve = max(8, min(48, animal_count * 4))
    orders = []
    shortage = max(0, reserve - carried_wheat - shed_wheat)
    if shortage > 0 and money > 500.0 and remaining > 24:
        orders.append(["BUY_PRODUCT", "WHEAT", min(16, shortage)])

    # Conservative expansion planner. It only creates inventory that the
    # learned PLANT_BEST / DEPLOY_ANIMAL skills can actually consume.
    crop_scores, animal_scores, _, _ = _strategy_scores(observation)
    remaining_days = max(0.0, remaining / 24.0)
    private = _private(observation)
    seeds = _get(private, "seeds", {}) or {}
    empty_tiles = sum(
        cell is None
        for row in (_get(farm, "tiles", []) or [])
        for cell in (row or [])
    )
    locked_tiles = sum(
        cell == "LOCKED"
        for row in (_get(farm, "tiles", []) or [])
        for cell in (row or [])
    )
    cash_reserve = max(1500.0, 35.0 * reserve)
    crop_meta = {
        "WHEAT": (10, 2, 6), "CARROT": (20, 2, 4),
        "TOMATO": (50, 8, 4), "STRAWBERRY": (100, 10, 4),
        "MELON": (80, 10, 6),
    }
    viable_crops = []
    for crop, (seed_cost, first_yield, max_yield) in crop_meta.items():
        if remaining_days < first_yield + 1.0:
            continue
        price = float(_get(prices, crop, base_price[crop]) or 0.0)
        gross_roi = (max_yield * price) / max(1.0, float(seed_cost))
        score = crop_scores[crop] + 0.08 * min(10.0, gross_roi)
        viable_crops.append((score, crop))
    if viable_crops and empty_tiles > 0 and money > cash_reserve + 200.0:
        _, best_crop = max(viable_crops)
        seed_stock = int(_get(seeds, best_crop, 0) or 0)
        if seed_stock < 3 and crop_scores[best_crop] >= 1.05:
            orders.append(["BUY_SEED", best_crop, min(2, empty_tiles)])

    animal_meta = {
        "GOOSE": (300, 4, 1, "EGG"),
        "COW": (400, 8, 2, "MILK"),
        "SHEEP": (500, 6, 3, "WOOL"),
    }
    active_animals = {"GOOSE": 0, "COW": 0, "SHEEP": 0}
    for row in _get(farm, "tiles", []) or []:
        for tile in row or []:
            if isinstance(tile, dict) and tile.get("animal") in active_animals:
                active_animals[tile["animal"]] += 1
    pending_animals = {
        animal: int(_get(shed, animal, 0) or 0)
        + sum(int(_get(inv, animal, 0) or 0) for inv in inventories)
        for animal in active_animals
    }
    viable_animals = []
    for animal, (cost, first_yield, interval, product) in animal_meta.items():
        if remaining_days < first_yield + 1.0:
            continue
        yields = 1 + int(max(0.0, remaining_days - first_yield) // interval)
        price = float(_get(prices, product, base_price[product]) or 0.0)
        demand_boost = 1.0 + 0.10 * demand[product]
        gross_roi = yields * price * demand_boost / max(1.0, float(cost))
        score = animal_scores[animal] + 0.30 * min(3.0, gross_roi)
        viable_animals.append((score, gross_roi, animal, cost))
    if viable_animals:
        score, gross_roi, best_animal, cost = max(viable_animals)
        total_type = active_animals[best_animal] + pending_animals[best_animal]
        total_pending = sum(pending_animals.values())
        if (
            score >= 1.25
            and gross_roi >= 0.95
            and total_type < 6
            and total_pending < 2
            and money > cash_reserve + cost
            and empty_tiles > 0
        ):
            orders.append(["BUY_ANIMAL", best_animal, 1])

    if empty_tiles == 0 and locked_tiles > 0 and remaining > 240 and money > 6000.0:
        orders.append(["BUY_LAND"])

    for item in ("CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "WHEAT"):
        qty = int(_get(shed, item, 0) or 0)
        if item == "WHEAT":
            qty = max(0, qty - reserve)
        if item == "FERTILIZER" and remaining > 72:
            qty = max(0, qty - 8)
        if qty <= 0:
            continue
        price = float(_get(prices, item, base_price[item]) or 0.0)
        ratio = price / max(1.0, base_price[item])
        pressure = rival_pressure[item]
        liquidate = remaining <= 48
        front_run = pressure >= 6.0 and ratio >= 0.90
        strong_price = ratio >= 1.12
        no_demand_profit = demand[item] <= 0.0 and ratio >= 1.05
        if liquidate or front_run or strong_price or no_demand_profit:
            orders.append(["SELL", item, qty])
    return orders


def _tiles(farm):
    return _get(farm, "tiles", []) or []


def farm_transition_reward(prev_obs, curr_obs, action, seat):
    """Dense shaping from outcomes, not merely requested actions."""
    prev_farms = list(_get(prev_obs, "farms", []) or [])
    curr_farms = list(_get(curr_obs, "farms", []) or [])
    if not (0 <= int(seat) < len(prev_farms) and 0 <= int(seat) < len(curr_farms)):
        return 0.0, {}
    prev, curr = prev_farms[int(seat)], curr_farms[int(seat)]
    reward = 0.0
    stats = {
        "plant_deaths": 0, "animal_escapes": 0,
        "watered": 0, "fed": 0, "cared": 0, "fertilized": 0,
        "fertilizer_collected": 0, "harvested_units": 0,
        "placed_units": 0, "sold_units": 0,
        "plants_created": 0, "structures_built": 0, "animals_placed": 0,
        "yield_gain": 0, "invalid_ops": 0,
        "critical_water_opportunities": 0, "water_attempts": 0, "water_success_actions": 0,
        "fert_collect_opportunities": 0, "fert_collect_attempts": 0, "fert_collect_success_actions": 0,
    }
    prev_tiles, curr_tiles = _tiles(prev), _tiles(curr)
    # Opportunity counters are diagnostic only: count asset-frames where an
    # operation is genuinely available before applying this turn's action.
    for row in prev_tiles:
        for tile in row or []:
            if not isinstance(tile, dict):
                continue
            if (
                tile.get("kind") == "PLANT"
                and not tile.get("watered_today", False)
                and int(tile.get("consecutive_unwatered", 0) or 0) >= 1
            ):
                stats["critical_water_opportunities"] += 1
            if "animal" in tile and bool(tile.get("fertilizer_available", False)):
                stats["fert_collect_opportunities"] += 1
    for y in range(min(len(prev_tiles), len(curr_tiles))):
        for x in range(min(len(prev_tiles[y]), len(curr_tiles[y]))):
            a, b = prev_tiles[y][x], curr_tiles[y][x]
            if a is None and isinstance(b, dict) and b.get("kind") == "PLANT":
                reward += 0.020
                stats["plants_created"] += 1
            if (
                a is None
                and isinstance(b, dict)
                and b.get("kind") in {"COOP", "PASTURE"}
                and "animal" not in b
            ):
                reward += 0.010
                stats["structures_built"] += 1
            if (
                isinstance(b, dict)
                and "animal" in b
                and not (isinstance(a, dict) and "animal" in a)
            ):
                reward += 0.040
                stats["animals_placed"] += 1
            if isinstance(a, dict) and a.get("kind") == "PLANT":
                if (
                    isinstance(b, dict) and b.get("kind") == "WEED"
                    and int(a.get("consecutive_unwatered", 0) or 0) >= 1
                    and not a.get("watered_today", False)
                ):
                    reward -= 1.5; stats["plant_deaths"] += 1
                elif isinstance(b, dict) and b.get("kind") == "PLANT":
                    if not a.get("watered_today", False) and b.get("watered_today", False): reward += 0.012; stats["watered"] += 1
                    if int(b.get("fertilized_until_day", -1) or -1) > int(a.get("fertilized_until_day", -1) or -1): reward += 0.015; stats["fertilized"] += 1
            if isinstance(a, dict) and "animal" in a:
                if (
                    not (isinstance(b, dict) and "animal" in b)
                    and int(a.get("consecutive_unfed", 0) or 0) >= 1
                    and not a.get("fed_today", False)
                ):
                    reward -= 4.0; stats["animal_escapes"] += 1
                else:
                    if not a.get("fed_today", False) and b.get("fed_today", False): reward += 0.018; stats["fed"] += 1
                    if not a.get("cared_today", False) and b.get("cared_today", False): reward += 0.008; stats["cared"] += 1

            if isinstance(a, dict) and isinstance(b, dict):
                gain = max(0, int(b.get("yield_units", 0) or 0) - int(a.get("yield_units", 0) or 0))
                if gain:
                    reward += min(0.05, 0.006 * gain)
                    stats["yield_gain"] += gain

    positions = _positions(prev)
    unit_actions = [_normalize_action((action or {}).get("farmer"))] + [
        _normalize_action(x) for x in ((action or {}).get("hands") or [])
    ]
    prev_invs = _inventories(prev_obs, len(positions))
    curr_invs = _inventories(curr_obs, len(positions))
    prev_shed = _shed(prev_obs)
    curr_shed = _shed(curr_obs)
    products = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
    for idx, requested in enumerate(unit_actions[:len(positions)]):
        op = str(requested[0])
        before_inv = prev_invs[idx] if idx < len(prev_invs) else {}
        after_inv = curr_invs[idx] if idx < len(curr_invs) else {}
        if op == "COLLECT_FERTILIZER":
            stats["fert_collect_attempts"] += 1
            delta = max(0, int(_get(after_inv, "FERTILIZER", 0) or 0) - int(_get(before_inv, "FERTILIZER", 0) or 0))
            if delta:
                reward += min(0.04, 0.012 * delta)
                stats["fertilizer_collected"] += delta
                stats["fert_collect_success_actions"] += 1
        elif op == "WATER":
            stats["water_attempts"] += 1
            # Existing tile-transition logic below still owns reward and total
            # watered count; this counter only measures requested execution.
            pos = positions[idx] if idx < len(positions) else None
            before_tile = _tile_at(prev, pos) if pos is not None else None
            after_tile = _tile_at(curr, pos) if pos is not None else None
            if (
                isinstance(before_tile, dict) and before_tile.get("kind") == "PLANT"
                and isinstance(after_tile, dict) and after_tile.get("kind") == "PLANT"
                and not before_tile.get("watered_today", False)
                and bool(after_tile.get("watered_today", False))
            ):
                stats["water_success_actions"] += 1
        elif op == "HARVEST":
            gained = sum(max(0, int(_get(after_inv, p, 0) or 0) - int(_get(before_inv, p, 0) or 0)) for p in products)
            if gained:
                reward += min(0.06, 0.008 * gained)
                stats["harvested_units"] += gained
        elif op == "PLACE" and len(requested) >= 2:
            item = str(requested[1])
            delta = max(0, int(_get(curr_shed, item, 0) or 0) - int(_get(prev_shed, item, 0) or 0))
            if delta:
                reward += min(0.04, 0.006 * delta)
                stats["placed_units"] += delta

    sell_items = {
        str(order[1]) for order in ((action or {}).get("market") or [])
        if isinstance(order, (list, tuple)) and len(order) >= 3 and str(order[0]) == "SELL"
    }
    for item in sell_items:
        delta = max(0, int(_get(prev_shed, item, 0) or 0) - int(_get(curr_shed, item, 0) or 0))
        if delta:
            reward += min(0.05, 0.004 * delta)
            stats["sold_units"] += delta

    for idx, requested in enumerate(unit_actions[:len(positions)]):
        op = str(requested[0])
        if op not in TILE_OPS:
            continue
        tile = _tile_at(prev, positions[idx])
        valid = True
        if op == "WATER": valid = isinstance(tile, dict) and tile.get("kind") == "PLANT" and not tile.get("watered_today", False)
        elif op in {"FEED", "CARE", "COLLECT_FERTILIZER"}: valid = isinstance(tile, dict) and "animal" in tile
        elif op == "FERTILIZE": valid = isinstance(tile, dict) and tile.get("kind") == "PLANT"
        elif op == "HARVEST": valid = isinstance(tile, dict) and int(tile.get("yield_units", 0) or 0) > 0
        elif op == "DIG": valid = tile is not None and not (isinstance(tile, dict) and "animal" in tile)
        if not valid:
            # Static macro routes intentionally contain harmless no-op retries.
            # Track them for diagnostics but do not let them dominate PPO.
            stats["invalid_ops"] += 1
    return float(reward), stats
