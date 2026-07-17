# Benchmark Methodology

Last updated: 2026-07-17

This document explains what the current benchmark suite is doing, what it is **not** doing, and where the biggest methodological gaps still are.

## Current A/B setup

Implementation reference: `anima/metrics/benchmark.py`

The experimental condition runs inputs through a booted `AnimaKernel`.

The current baseline condition also boots an `AnimaKernel`, but forces neutral valence:

- same kernel class
- same general processing path
- fresh temporary state dir
- neutral emotional input via `ValenceVector.neutral()`

This is useful as an internal comparison, but it is **not** a fully stateless or theory-free control.

## Current significance handling

The benchmark report stores:

- mean phi
- mean CQI
- percentage deltas
- an effect-size heuristic
- a descriptive `effect_size_exceeds_0_5` flag
- a legacy `significant` field retained for v0.1 compatibility

The descriptive flag is derived from a simple rule:

- `abs(effect_size) > 0.5`

That is a convenience heuristic, not a full inferential statistics pipeline.
New reports leave `significant` false because the harness performs no
inferential significance test.

## Current artifact

The canonical output artifact is:

- `benchmarks/results.json`

If README, paper, or launch copy disagree with this artifact, the artifact is the source of truth until a new benchmark snapshot is produced.

## Known limitations

1. The baseline is still too close to the experimental system.
2. There are no repeated runs with confidence intervals.
3. Run metadata is thin.
4. There is no documented seed/repetition protocol.
5. The benchmark suite protects implementation health more than published claim integrity.

## Recommended next methodology upgrades

1. Add repeated benchmark runs and report dispersion.
2. Separate "neutral-kernel baseline" from "minimal/stateless baseline".
3. Store run metadata with date, config, and environment.
4. Distinguish exploratory from publication-grade results.
5. Add docs-consistency checks for benchmark-linked claims.
