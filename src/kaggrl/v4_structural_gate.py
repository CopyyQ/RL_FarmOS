import json
from dataclasses import dataclass
from pathlib import Path


COW_BRANCH_ROUTE = 105
COW_BRANCH_START_STEP = 144
COW_BRANCH_HORIZON = 48


def _load_gain_table():
    root = Path(__file__).resolve().parents[2]
    path = root / "assets" / "structural_gain_table_v4_2.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "farmos_structural_gain_table_v4_2":
            return {}, frozenset()
        routes = frozenset(int(x) for x in payload.get("source_routes", []))
        return payload, routes
    except Exception:
        # Safe fallback: never apply an unverified structural override.
        return {}, frozenset()


STRUCTURAL_GAIN_TABLE, COW_BRANCH_SOURCE_ROUTES = _load_gain_table()


@dataclass(frozen=True)
class StructuralOverride:
    route_id: int
    until_step: int
    reason: str


def structural_override_for_state(
    *,
    step: int,
    base_route_id: int,
) -> StructuralOverride | None:
    if (
        int(step) == COW_BRANCH_START_STEP
        and int(base_route_id) in COW_BRANCH_SOURCE_ROUTES
    ):
        stats = (
            STRUCTURAL_GAIN_TABLE.get("routes", {})
            .get(str(int(base_route_id)), {})
        )
        mean_gain = float(stats.get("mean_gain", 0.0))
        samples = int(stats.get("n", 0))
        return StructuralOverride(
            route_id=COW_BRANCH_ROUTE,
            until_step=COW_BRANCH_START_STEP + COW_BRANCH_HORIZON,
            reason=(
                "exact_gain_table_v4_2:"
                f"base={int(base_route_id)}:"
                f"n={samples}:mean_gain={mean_gain:.1f}"
            ),
        )
    return None
