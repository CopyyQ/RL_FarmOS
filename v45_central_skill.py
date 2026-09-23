from __future__ import annotations

import kaggrl.v4_farm_supervisor as fs

SALE_ITEMS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL")


def _inventory_total(inv):
    return sum(max(0, int(v or 0)) for v in (inv or {}).values())

def plan_state_aware_skills(observation, remaining_steps=999):
    farm, _ = fs._farm(observation)
    if farm is None:
        return {"farmer":["PASS"],"hands":[],"market":[]}, {"active":False}
    positions = fs._positions(farm)
    hand_count = max(0, len(positions)-1)
    invs = fs._inventories(observation, len(positions))
    shed = fs._shed(observation)
    day = int(fs._get(observation, "day", 0) or 0)
    targets = fs._all_targets(farm, day)
    out = {"farmer":["PASS"], "hands":[["PASS"] for _ in range(hand_count)], "market":[]}
    assigned = set()
    claimed = set()
    counts = {}

    def set_action(unit_idx, action, kind):
        if unit_idx == 0:
            out["farmer"] = list(action)
        else:
            out["hands"][unit_idx-1] = list(action)
        assigned.add(unit_idx)
        counts[kind] = counts.get(kind, 0) + 1
    def assign(target_list, kind, op, capability=None):
        for target in target_list:
            target = tuple(target)
            if target in claimed:
                continue
            candidates = []
            for unit_idx, pos in enumerate(positions):
                if unit_idx in assigned:
                    continue
                inv = invs[unit_idx] if unit_idx < len(invs) else {}
                if capability is not None and not capability(inv):
                    continue
                candidates.append((fs._distance(pos, target), unit_idx, pos))
            if not candidates:
                continue
            _, unit_idx, pos = min(candidates)
            claimed.add(target)
            if tuple(pos) == target:
                set_action(unit_idx, [op], kind)
            else:
                set_action(unit_idx, fs._step_toward(pos, target), "move")

    assign(targets.get("feed", []), "feed", "FEED", lambda inv: int(fs._get(inv,"WHEAT",0) or 0) > 0)
    assign(targets.get("water", []), "water", "WATER")
    assign(targets.get("harvest", []), "harvest", "HARVEST")
    assign(targets.get("care", []), "care", "CARE")
    board = max(1, len(fs._get(farm, "tiles", []) or []))
    access_tiles = fs._shed_access(board)
    for unit_idx, pos in enumerate(positions):
        if unit_idx in assigned:
            continue
        inv = invs[unit_idx] if unit_idx < len(invs) else {}
        if _inventory_total(inv) <= 0:
            continue
