# Contributing to TurboQuant

## Scope

TurboQuant contributions should remain focused on:

- quantization correctness and numerical stability,
- measurable compression/runtime improvements,
- reproducible benchmarking and clear regression signals.

## Minimum Validation Before Sending a Change

1. `python -m py_compile` on modified TurboQuant Python modules.
2. TurboQuant unit tests:
   - `//tensorflow/python/ops/turboquant:core_test`
   - `//tensorflow/python/ops/turboquant:keras_test`
   - `//tensorflow/python/ops/turboquant:calibration_test`
   - `//tensorflow/python/ops/turboquant:benchmark_test`
3. Smoke benchmark:
   - `//tensorflow/python/ops/turboquant:benchmark_turboquant`

## Performance Changes

- Include before/after benchmark JSON artifacts.
- Report at least 3 seeds for any claimed quality/performance improvement.
- Keep configuration and hardware details in review notes.

## Serialization Changes

- Preserve backward compatibility for existing `format_version` payloads.
- Add explicit tests for new fields and failure modes.
