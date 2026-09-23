from dataclasses import dataclass
from dataclasses import dataclass
from typing import Any


MOVES = frozenset({"NORTH", "SOUTH", "EAST", "WEST"})
TILE_OPS = frozenset({
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "DIG",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
    "PLACE",
})


@dataclass(frozen=True)
class LaborOptimization:
    applied: bool
    weeds_dug: int
    invalid_digs_removed: int
    unit_changes: tuple[tuple[int, tuple, tuple], ...]


def _tile_at(farm: dict[str, Any], pos):
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return None
    x, y = int(pos[0]), int(pos[1])
    tiles = farm.get("tiles") or []
    if not (0 <= y < len(tiles)):
        return None
    row = tiles[y]
    if not (0 <= x < len(row)):
        return None
    return row[x]


def _is_weed(tile) -> bool:
    return isinstance(tile, dict) and tile.get("kind") == "WEED"


def _normalize_action(raw):
    if isinstance(raw, list) and raw:
        return list(raw)
    return ["PASS"]


def optimize_idle_weed_labor(
    observation: dict[str, Any],
    action: dict[str, Any],
) -> tuple[dict[str, Any], LaborOptimization]:
    """Use otherwise wasted unit actions to clear weeds.

    Safety rule:
    - never override movement;
    - never override an action that can be useful on the current non-weed tile;
    - on a WEED tile, replace PASS or tile operations that are guaranteed to
      no-op on WEED with DIG;
    - replace DIG on an already-empty tile with PASS.

    This does not hire workers or alter market orders. It only recovers labor
    that the macro schedule would otherwise waste because the stochastic weed
    state differs from the static route assumption.
    """
    out = {
        "farmer": _normalize_action((action or {}).get("farmer")),
        "hands": [
            _normalize_action(x) for x in ((action or {}).get("hands") or [])
        ],
        "market": [
            list(x) for x in ((action or {}).get("market") or [])
            if isinstance(x, (list, tuple))
        ],
    }
    player = int(observation.get("player", 0))
    farms = observation.get("farms") or []
    if not (0 <= player < len(farms)):
        return out, LaborOptimization(False, 0, 0, ())
    farm = farms[player]
    positions = [farm.get("farmer")] + list(farm.get("hands") or [])

    while len(out["hands"]) < max(0, len(positions) - 1):
        out["hands"].append(["PASS"])

    changes = []
    weeds_dug = 0
    invalid_digs_removed = 0
    for unit_idx, pos in enumerate(positions):
        planned = out["farmer"] if unit_idx == 0 else out["hands"][unit_idx - 1]
        planned = _normalize_action(planned)
        op = str(planned[0])
        tile = _tile_at(farm, pos)

        replacement = None
        if _is_weed(tile):
            # Every tile mutation except DIG is invalid on WEED. Movement is
            # intentionally preserved because the unit may be travelling to a
            # more valuable task.
            if op == "PASS" or (op in TILE_OPS and op != "DIG"):
                replacement = ["DIG"]
                weeds_dug += 1
        elif tile is None and op == "DIG":
            replacement = ["PASS"]
            invalid_digs_removed += 1

        if replacement is not None and replacement != planned:
            changes.append(
                (unit_idx, tuple(planned), tuple(replacement))
            )
            if unit_idx == 0:
                out["farmer"] = replacement
            else:
                out["hands"][unit_idx - 1] = replacement

    return out, LaborOptimization(
        applied=bool(changes),
        weeds_dug=int(weeds_dug),
        invalid_digs_removed=int(invalid_digs_removed),
        unit_changes=tuple(changes),
    )


@dataclass(frozen=True)
class HireOptimization:
    applied: bool
    hires_removed: int
    hires_today_before: int
    remaining_steps: int


def optimize_late_hires(
    observation: dict[str, Any],
    action: dict[str, Any],
    *,
    remaining_steps: int,
    max_hires_today: int = 9,
    late_window_steps: int = 48,
) -> tuple[dict[str, Any], HireOptimization]:
    """Prune only hires beyond the validated late-game labor cap.

    Counterfactual exact games showed that the 10th daily hire can become
    negative-value in the final two days. This gate is intentionally narrow:
    it does not touch early/mid-game labor or the first nine hires.
    """
    out = {
        "farmer": _normalize_action((action or {}).get("farmer")),
        "hands": [
            _normalize_action(x) for x in ((action or {}).get("hands") or [])
        ],
        "market": [
            list(x) for x in ((action or {}).get("market") or [])
            if isinstance(x, (list, tuple))
        ],
    }
    player = int(observation.get("player", 0))
    farms = observation.get("farms") or []
    if not (0 <= player < len(farms)):
        return out, HireOptimization(
            False, 0, 0, int(remaining_steps)
        )

    farm = farms[player]
    hires_before = int(farm.get("hires_today", 0))
    if (
        int(remaining_steps) > int(late_window_steps)
        or hires_before < int(max_hires_today)
    ):
        return out, HireOptimization(
            False, 0, hires_before, int(remaining_steps)
        )

    kept = []
    removed = 0
    running_hires = hires_before
    for order in out["market"]:
        if order and str(order[0]) == "HIRE":
            if running_hires >= int(max_hires_today):
                removed += 1
                continue
            running_hires += 1
        kept.append(order)
    out["market"] = kept
    return out, HireOptimization(
        applied=removed > 0,
        hires_removed=int(removed),
        hires_today_before=hires_before,
        remaining_steps=int(remaining_steps),
    )
