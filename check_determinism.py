import hashlib
import json
from pathlib import Path

from winner_train import _run_chunk

ROOT = Path(__file__).resolve().parent


def sig(records):
    payload = [
        (
            int(r["step"]),
            int(r["route_action"]),
            int(r["market_action"]),
            int(r["horizon_action"]),
            str(r["market_mode"]),
            int(r["chosen_route_id"]),
        )
        for r in records
    ]
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def main():
    args = (
        str(ROOT / "assets" / "parent_promoted_v2.pt"),
        str(ROOT / "output" / "winner_v4_latest.pt"),
        str(ROOT / "assets" / "v51_main.py"),
        [(22200487, 0)] * 5,
        False,
        0.90,
        1.0,
        2.3,
        24,
        999,
    )
    rows = _run_chunk(args)
    for i, row in enumerate(rows):
        print(
            i,
            row["own_money"],
            row["v51_money"],
            row["margin"],
            len(row["records"]),
            sig(row["records"]),
            flush=True,
        )


if __name__ == "__main__":
    main()
