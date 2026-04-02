# Fork Notes

This document tracks fork-specific work that is not part of upstream
TensorFlow release notes.

## Attribution

TurboQuant extension in this fork: **Shoko**

Primary implementation area:
- `tensorflow/python/ops/turboquant/`

## Fork Changelog (TurboQuant)

### 2026-03-28

- Added and stabilized initial TurboQuant implementation across core packing,
  Keras integration, calibration flows, and test scaffolding.
- Added activation-aware calibration thresholds and coverage.

### 2026-03-29

- Added scoped quantization controls and aggregate reporting.
- Hardened benchmark outputs and Bazel target integration.
- Expanded tests for calibration reports and selection filters.
- Fixed Bazel visibility for TurboQuant Keras dependencies.

### 2026-03-30

- Optimized core codebook assignment path.
- Reduced packed index memory footprint in Keras wrapper state.
- Added benchmark matrix tooling (`p50/p95`, throughput, drift, compression).
- Added profiling utility for stage-level timings.

### 2026-03-31

- Documented reproducible benchmark workflow and reporting protocol.

### 2026-04-01

- Tightened per-layer auto-tuning candidate search logic and reporting.

### 2026-04-02

- Added compact core serialization helpers and packed-index utilities.
- Added real-model benchmark runner and scientific analysis tooling
  (multi-seed aggregation and ablation scripts).
- Added dedicated TurboQuant CI workflow and release governance docs.
- Added versioned serialization contract with compatibility checks.
- Added experimental C++ custom op for packed-index unpacking.

## Related Fork Docs

- `tensorflow/python/ops/turboquant/README.md`
- `tensorflow/python/ops/turboquant/REPRODUCIBILITY.md`
- `tensorflow/python/ops/turboquant/CONTRIBUTING_TURBOQUANT.md`
- `tensorflow/python/ops/turboquant/ROADMAP.md`
- `tensorflow/python/ops/turboquant/KNOWN_LIMITATIONS.md`
- `tensorflow/python/ops/turboquant/RELEASE_CHECKLIST.md`
