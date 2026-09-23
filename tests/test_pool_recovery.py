import multiprocessing as mp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src"), str(ROOT)]

from winner_train import ResilientPool


def _square(value):
    return value * value


def test_pool_recovery():
    ctx = mp.get_context("spawn")
    pool = ResilientPool(
        ctx,
        processes=2,
        maxtasksperchild=8,
        timeout_seconds=10.0,
        recovery_retries=1,
    )
    try:
        assert pool.map(_square, [1, 2, 3]) == [1, 4, 9]
        pool.pool.terminate()
        pool.pool.join()
        assert pool.map(_square, [4, 5]) == [16, 25]
    finally:
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    test_pool_recovery()
    print("POOL_RECOVERY_OK")
