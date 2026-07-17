# ANIMA Benchmark Evidence Status

Last updated: 2026-07-17

This file is the plain-language companion to `benchmarks/results.json`.
It exists to keep the benchmark story honest.

## What is currently strong

- The kernel is real and testable.
- Local test run on 2026-07-17: `446 passed`.
- The benchmark suite exists and produces a reproducible JSON artifact.
- Working-memory ablation shows a strong structural effect in the current snapshot.

## What the current benchmark snapshot says

Source of truth: `benchmarks/results.json`

- `overall_improvement_pct`: `0.92`
- All 5 A/B comparisons have descriptive effect sizes below `0.5`
- Current valence ablation impact: `2.02%` CQI
- Current temporal ablation impact: `0.0%` CQI
- Working-memory ablation remains the strongest effect in the saved snapshot

## What this means

ANIMA is already compelling as:

- an architectural proposal
- a clean kernel implementation
- an instrumented cognitive-state engine

ANIMA is **not yet** on equally strong ground as a settled empirical claim package,
and its metrics are not evidence of phenomenal consciousness.

## Important caution

Some draft/public-facing materials describe stronger numbers than the current saved benchmark artifact supports.

When there is a mismatch:

1. `benchmarks/results.json` wins
2. stronger claims must be marked as historical, draft, or outdated
3. new benchmark claims should link to the exact artifact used

## Next upgrades

- align paper and launch copy with current saved results
- improve baseline design
- add repetitions, run metadata, and better significance handling
- keep this file updated whenever benchmark methodology changes
