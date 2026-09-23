import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src"), str(ROOT)]

from winner_train import _run_chunk

args = (
    str(ROOT / "assets" / "parent_promoted_v2.pt"),
    os.environ.get(
        "FARMOS_CHECKPOINT",
        str(ROOT / "runs" / "v46_econ_shadow" / "winner_v45_skill_stage_safe.pt"),
    ),
    str(ROOT / "assets" / "v51_main.py"),
    [(22200487, 0)],
    False,
    0.90,
    1.0,
    2.3,
    24,
    999,
)
row = _run_chunk(args)[0]
trace = [
    (
        int(r["step"]),
        int(r["route_action"]),
        int(r["market_action"]),
        int(r["horizon_action"]),
        str(r["market_mode"]),
        int(r["chosen_route_id"]),
    )
    for r in row["records"]
]
digest = hashlib.sha256(
    json.dumps(trace, separators=(",", ":")).encode()
).hexdigest()[:16]
print(row["own_money"], row["v51_money"], row["margin"], digest)
