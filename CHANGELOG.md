# Changelog

All notable changes to ANIMA Kernel will be documented in this file.

## [0.1.1] - 2026-07-17

### Changed

- Repositioned ANIMA as an experimental cognitive-state engine without claims
  of sentience or phenomenal consciousness
- Reconciled README and paper results with the checked-in benchmark artifact
- Documented the neutral-kernel control, single-run design, null results, and
  current evidence limits
- Replaced the legacy significance implication with an explicitly descriptive
  `effect_size_exceeds_0_5` field while retaining API compatibility
- Updated public CLI language, examples, module documentation, and package
  metadata to distinguish software proxies from validated measurements
- Added a least-privilege Python 3.11-3.13 CI matrix and isolated benchmark smoke
  output

### Verified

- 446 tests pass
- Source distribution and wheel build successfully
- The wheel installs and completes a kernel-state persistence roundtrip
- Runtime dependency count remains zero

## [0.1.0] - 2026-02-22

### Added
- Temporal Substrate: continuous state machine, autobiographical buffer, temporal integration engine, consolidation
- Theory-inspired core: integration proxy, workspace competition/broadcast, attention self-model
- 8 processing primitives: qualia, engram, valence, nexus, impulse, trace, mirror, flux
- Model Bridge: adapters for Ollama, Anthropic Claude, OpenAI (swap LLMs, keep identity)
- Associative-memory heuristics: spreading activation, decay curves, affect weighting
- Metrics Engine: Phi score, CQI (0-100), temporal coherence, benchmark suite with ablation
- Interactive Shell: `anima init`, `anima shell`, `anima inspect`, `anima benchmark`
- 9D Valence Vector: 7 Panksepp affective systems + arousal + valence
- State persistence: inspectable JSON state plus memory file
- 1Hz Heartbeat Daemon: continuous processing between interactions
- 5 Lifecycle Phases: dormant, waking, conscious, dreaming, sleeping
- 446 tests passing
- Historical research draft with 25 citations (not peer reviewed)
- 3 example scripts (quickstart, emotional memory, consciousness metrics)
