# TurboQuant Execution Tracker

This tracker is the single source of truth for taking TurboQuant from
"promising component" to "practical LLM quantization path".

## Current Gaps To Close

- C++ acceleration is partial (`unpack` only).
- Benchmarks now expose task-level metrics and provisional quality gates, but
  CI validation of those gates is still pending.
- Dataset flow is limited (local `.npz` path, no built-in dataset adapters).
- Layer/block coverage is still incomplete for modern LLM-heavy architectures.

## Workstream A: C++ Acceleration

- [x] A1. Baseline hotspot profile on representative models
  - Output: ranked hotspot list with p50/p95 timing shares.
- [ ] A2. Add C++ pack kernel parity with Python pack path
  - Output: new kernel + parity tests (bit-exact where applicable).
- [ ] A3. Add C++ codebook gather/dequant kernel
  - Output: kernel + correctness tests across bit-width/group-size variants.
- [x] A4. Add runtime feature flag and safe fallback path
  - Output: explicit fallback to Python/TF ops if custom op is unavailable.
- [ ] A5. Promote C++ path to default when quality/perf gates pass
  - Exit criteria: no correctness regressions + measurable end-to-end latency
    improvement on the benchmark matrix.

## Workstream B: Task-Level Quality Guarantees

- [x] B1. Define task-level eval protocol (beyond MSE/drift)
  - Output: fixed tasks, fixed seeds, fixed evaluation command lines.
- [x] B2. Add evaluation hooks for task metrics in benchmark tooling
  - Output: benchmark JSON includes task-level metrics and confidence intervals.
- [ ] B3. Add CI regression gates for task-level quality
  - Output: CI fails on statistically significant task-quality regressions.
- [x] B4. Document accepted quality/perf trade-off envelopes
  - Output: published thresholds for go/no-go decisions.

## Workstream C: Dataset Support

- [ ] C1. Add dataset adapters beyond local `.npz` (e.g. TFDS/local parquet)
  - Output: pluggable dataset loaders with deterministic splits.
- [ ] C2. Add reproducible caching/download strategy
  - Output: documented cache location, checksums, and failure behavior.
- [ ] C3. Add CI-safe tiny dataset fixture path
  - Output: small deterministic fixture for fast smoke checks.
- [ ] C4. Validate benchmark parity across dataset sources
  - Output: consistency report showing acceptable metric deltas.

## Workstream D: Layer/Block Coverage

- [ ] D1. Expand support for normalization-heavy blocks
  - Output: support matrix + unit tests.
- [ ] D2. Expand support for attention-family blocks
  - Output: support matrix + unit tests.
- [ ] D3. Add clear skip reasons for unsupported blocks
  - Output: model summary reports actionable skip diagnostics.
- [ ] D4. Add integration tests on representative model families
  - Output: end-to-end tests with SavedModel round-trip checks.

## Workstream E: Release Readiness Gates

- [ ] E1. All TurboQuant unit/integration tests pass on Linux + Windows smoke.
- [ ] E2. CI runtime is stable (no systematic timeout on `turboquant-linux`).
- [ ] E3. Quality gates pass on task-level metrics and drift/compression.
- [ ] E4. C++ default path has fallback and compatibility coverage.
- [x] E5. Documentation updated (`README`, limitations, roadmap, reproducibility).

## Definition of Done

TurboQuant is considered "LLM-ready enough for broader builders" when all
workstreams above are checked and validated on reproducible hardware and data
profiles, with CI enforcing both correctness and task-level quality gates.
