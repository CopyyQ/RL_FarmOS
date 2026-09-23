# FarmOS Candidate Tournament

CPU-first successive-halving harness for selecting policy/runtime candidates.

## Workflow

1. Generate or provide a candidate manifest.
2. Run a small representative screening stage.
3. Keep only the strongest non-regressing candidates.
4. Race survivors on fresh seeds.
5. Validate finalists on larger fresh sets.
6. Use a final holdout before promotion.

The runner caches compact per-game results by checkpoint/config/seed/seat, so rerunning a stage does not recompute completed games.

## Default 100-candidate pilot

`configs/tournament/runtime100.json` contains 100 deterministic combinations of:

- ACT/KEEP threshold,
- skill confidence threshold,
- skill cutover step.

This manifest primarily validates the tournament infrastructure. Future manifests should add portfolio, market-timing, and risk-gated candidates as those runtime options become available.

## Commands

Generate the default 100 candidates:

```bash
python tools/tournament/generate_runtime_candidates.py
```

Benchmark worker count:

```bash
python tools/tournament/benchmark_workers.py --workers 4,8,12,16
```

Run screening only:

```bash
python tools/tournament/run.py --stage screen --workers 12
```

Run all stages:

```bash
python tools/tournament/run.py --workers 12
```

For a fast harness smoke test:

```bash
python tools/tournament/run.py --stage screen --limit-candidates 3 --workers 4
```

## Promotion rule

Do not promote a candidate from the screening stage alone. A policy candidate must survive fresh paired validation and tail-risk gates before merging into `farmosv1`.

See `knowledge/START_HERE.md` for the current benchmark baseline and promotion criteria.
