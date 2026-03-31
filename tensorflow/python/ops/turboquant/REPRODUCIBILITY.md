# TurboQuant Reproducibility Notes

This guide captures the benchmark protocol used for TurboQuant comparisons.

## Environment

- Use a fixed Python/TensorFlow build for the entire run.
- Pin CPU governor/performance mode when possible.
- Avoid concurrent heavy processes on the benchmark host.

## Recommended Commands

Benchmark suite:

```bash
python tensorflow/python/ops/turboquant/benchmark_turboquant.py \
  --seed 123 \
  --models dense_mlp,conv2d_stack,depthwise_separable \
  --batch_sizes 1,8,32 \
  --warmup_steps 10 \
  --benchmark_steps 50 \
  --repeats 3 \
  --num_bits 4 \
  --group_size 8 \
  --outlier_threshold 6.0 \
  --json_output /tmp/turboquant_benchmark.json
```

Stage-level profile:

```bash
python tensorflow/python/ops/turboquant/profile_turboquant.py \
  --seed 123 \
  --batch_size 16 \
  --warmup_steps 10 \
  --benchmark_steps 50 \
  --num_bits 4 \
  --group_size 8 \
  --outlier_threshold 6.0 \
  --json_output /tmp/turboquant_profile.json
```

## Metrics

- Compression:
  - `effective_ratio = total_original_bytes / total_packed_bytes`
- Accuracy drift:
  - `mean_squared_error`
  - `max_abs_error`
- Runtime:
  - latency `mean`, `p50`, `p95`
  - throughput `mean`, `p50`, `p95`

## Reporting

- Run each benchmark case with at least 3 repeats.
- Compare both aggregate stats and per-case outputs.
- Keep raw JSON artifacts with seed/config metadata to make comparisons
  auditable.
