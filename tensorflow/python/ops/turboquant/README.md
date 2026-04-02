# TurboQuant

TurboQuant is a lightweight weight-only quantization path for TensorFlow Python
layers built around three conservative ideas:

- per-output-channel codebooks instead of one global codebook,
- block-wise scales to preserve local dynamic range,
- optional outlier residuals kept in full precision for stability.

The current integration targets `tf.keras.layers.Dense`,
`tf.keras.layers.Conv1D`, `tf.keras.layers.Conv2D`,
`tf.keras.layers.Conv3D`, `tf.keras.layers.Conv1DTranspose`,
`tf.keras.layers.Conv2DTranspose`, `tf.keras.layers.Conv3DTranspose`,
`tf.keras.layers.DepthwiseConv2D`,
`tf.keras.layers.SeparableConv1D`, `tf.keras.layers.SeparableConv2D`, and
`tf.keras.layers.Embedding`
through `TurboDense`, `TurboConv1D`, `TurboConv2D`, `TurboConv3D`,
`TurboConv1DTranspose`, `TurboConv2DTranspose`, `TurboConv3DTranspose`,
`TurboDepthwiseConv2D`, `TurboSeparableConv1D`, `TurboSeparableConv2D`,
`TurboEmbedding`, and `quantize_model()` cloning helpers.
The core packing logic is kept in a NumPy-only module so the quantizer is easy
to test and extend before moving deeper into TensorFlow compiler paths.

## Example

```python
from tensorflow.python.ops.turboquant import api
from tensorflow.python.ops.turboquant import config

quantized_model = api.quantize_model(
    model,
    config.TurboQuantConfig(
        num_bits=4,
        group_size=64,
        outlier_threshold=6.0,
    ),
)

calibration_stats = api.collect_calibration_stats(
    model,
    representative_dataset,
    config.CalibrationConfig(max_steps=16, max_samples=2048),
)

quantized_model = api.quantize_model(
    model,
    config.TurboQuantConfig(
        num_bits=4,
        group_size=64,
        max_normalized_mean_squared_error=0.05,
        max_normalized_max_abs_error=0.25,
    ),
    representative_dataset=representative_dataset,
    calibration_config=config.CalibrationConfig(max_steps=16, max_samples=2048),
)

recommendations = api.recommend_layer_configs(
    model,
    config.TurboQuantConfig(num_bits=4, group_size=64),
    representative_dataset=representative_dataset,
    calibration_config=config.CalibrationConfig(max_steps=16, max_samples=2048),
    target_normalized_mean_squared_error=0.05,
    target_compression_ratio=1.5,
)

report = api.quantize_model(
    model,
    config.TurboQuantConfig(num_bits=4, group_size=64),
    auto_tune=True,
    target_normalized_mean_squared_error=0.05,
    target_compression_ratio=1.5,
    representative_dataset=representative_dataset,
    dry_run=True,
)

scoped_quantized_model = api.quantize_model(
    model,
    config.TurboQuantConfig(num_bits=4, group_size=64),
    target_layer_names=['encoder_dense', 'decoder_dense'],
    exclude_layer_names=['decoder_dense'],
)

summary_report = api.summarize_model(
    model,
    config.TurboQuantConfig(num_bits=4, group_size=64),
    include_skipped=True,
    return_report=True,
)
```

## Design Notes

- The first implementation is intentionally weight-only and inference-oriented.
  It avoids changing TensorFlow MLIR or SavedModel quantization passes until the
  Python-layer behavior and error profile are covered by tests.
- Packed indices use an unsigned compact layout (`uint8` up to 8-bit
  quantization levels) in both core encodings and Keras wrapper state to reduce
  runtime memory overhead.
- Compression numbers reported by `summarize_encoding()` and `summarize_model()`
  estimate the effective packed footprint. They are not raw checkpoint sizes.
- `serialize_encoding()` / `deserialize_encoding()` provide a compact
  bit-packed index format for storing or moving core TurboQuant encodings.
- Serialized payloads carry a format marker and version field to support
  compatibility checks during restoration.
- `summarize_model(include_skipped=True)` also reports ignored layers and the
  reason they were left untouched, such as a kernel that is too small or a
  packing layout that is not profitable.
- Separable convolutions are summarized component-wise so the report keeps both
  the depthwise and pointwise compression/error profile visible.
- `collect_calibration_stats()` gathers per-layer activation statistics from a
  representative dataset, and `summarize_model(..., calibration_stats=...)`
  adds normalized error metrics against observed activation scales.
- `recommend_layer_configs()` runs a constrained candidate search per layer and
  returns layer-specific TurboQuant settings that best meet compression and
  normalized-error objectives.
- `quantize_model(auto_tune=True, ...)` can consume the same objectives and
  automatically apply per-layer quantization settings.
- `target_layer_names` and `exclude_layer_names` allow explicit scoping of
  quantization/recommendation to specific layers in larger models.
- `quantize_model(..., representative_dataset=..., calibration_config=...)`
  can apply optional activation-aware skip heuristics when
  `max_normalized_mean_squared_error` or `max_normalized_max_abs_error` are set
  in `TurboQuantConfig`.
- `quantize_model(dry_run=True)` returns a structured decision report without
  cloning the model, and `strict=True` turns skipped selected layers into
  explicit errors.
- `quantize_model(return_report=True)` and `summarize_model(return_report=True)`
  include an aggregate model-level section with total packed bytes, effective
  compression ratio, and skip reason counts.
- Re-quantizing an already TurboQuant-wrapped model is idempotent: packed
  wrapper state is preserved rather than being interpreted as raw float weights.
- `export_saved_model()` emits a TensorFlow SavedModel with a stable
  `serving_default` signature for inference, and `load_saved_model()` restores
  the exported inference object through the core SavedModel loader.
- The tensor packing API is shape-generic, so extending support beyond the
  current wrappers does not require reworking the quantizer itself.

## Benchmark Suite

A reproducible benchmark suite is available in
`tensorflow/python/ops/turboquant/benchmark_turboquant.py`. It reports:

- matrix results across multiple model families and batch sizes,
- compression ratio statistics,
- output drift (MSE / max absolute error),
- latency and throughput distributions (`mean`, `p50`, `p95`) for float and
  TurboQuant variants.

The benchmark protocol used for comparisons is documented in
`tensorflow/python/ops/turboquant/REPRODUCIBILITY.md`.

Project maintenance references:
- `CONTRIBUTING_TURBOQUANT.md`
- `KNOWN_LIMITATIONS.md`
- `ROADMAP.md`
- `RELEASE_CHECKLIST.md`

Run:

```bash
python tensorflow/python/ops/turboquant/benchmark_turboquant.py \
  --models dense_mlp,conv2d_stack,depthwise_separable \
  --batch_sizes 1,8,32 \
  --repeats 3 \
  --json_output /tmp/turboquant_benchmark.json
```

Or through Bazel:

```bash
bazel run //tensorflow/python/ops/turboquant:benchmark_turboquant -- \
  --models=dense_mlp,conv2d_stack,depthwise_separable \
  --batch_sizes=1,8,32 \
  --repeats=3 \
  --json_output=/tmp/turboquant_benchmark.json
```

## Real-Model Benchmark

`benchmark_real_models.py` targets realistic CNN families and supports either
synthetic data or an external `.npz` dataset (`x` and optional `y` arrays):

```bash
python tensorflow/python/ops/turboquant/benchmark_real_models.py \
  --models residual_cnn,separable_cnn \
  --seeds 123,456,789 \
  --dataset_source synthetic \
  --train_epochs 1 \
  --json_output /tmp/turboquant_real_models.json
```

To benchmark on a local real dataset dump:

```bash
python tensorflow/python/ops/turboquant/benchmark_real_models.py \
  --dataset_source npz \
  --dataset_npz_path /path/to/dataset.npz \
  --json_output /tmp/turboquant_real_models_npz.json
```

`dataset.npz` must provide:
- `x`: float32 tensor shaped `[N, 32, 32, 3]`
- `y` (optional): int labels shaped `[N]`

## Scientific Analysis

Aggregate multiple benchmark reports with confidence intervals:

```bash
python tensorflow/python/ops/turboquant/analyze_turboquant_results.py \
  /tmp/turboquant_benchmark_seed1.json \
  /tmp/turboquant_benchmark_seed2.json \
  --json_output /tmp/turboquant_summary.json
```

Run an ablation search over quantization hyper-parameters:

```bash
python tensorflow/python/ops/turboquant/run_turboquant_ablations.py \
  --num_bits 2,3,4 \
  --group_sizes 8,16,32 \
  --outlier_thresholds 4.0,6.0,8.0 \
  --seeds 123,456,789 \
  --json_output /tmp/turboquant_ablations.json
```

## Profiling

For a stage-level profile (`summarize_model`, `quantize_model`, float
inference, TurboQuant inference), use:

```bash
python tensorflow/python/ops/turboquant/profile_turboquant.py \
  --batch_size 16 \
  --benchmark_steps 50 \
  --json_output /tmp/turboquant_profile.json
```

or with Bazel:

```bash
bazel run //tensorflow/python/ops/turboquant:profile_turboquant -- \
  --batch_size=16 \
  --benchmark_steps=50 \
  --json_output=/tmp/turboquant_profile.json
```

## Experimental C++ Path

An experimental C++ custom op (`TurboQuantUnpackIndices`) is provided for
packed-index unpacking in latency-sensitive deployments. The op is exposed via
`cpp_ops.py` and can be probed with `has_cpp_kernels()`.
