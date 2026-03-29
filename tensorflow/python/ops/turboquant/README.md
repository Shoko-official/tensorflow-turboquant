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
- Compression numbers reported by `summarize_encoding()` and `summarize_model()`
  estimate the effective packed footprint. They are not raw checkpoint sizes.
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

## Benchmark

A minimal reproducible benchmark is available in
`tensorflow/python/ops/turboquant/benchmark_turboquant.py`. It reports:

- per-layer effective compression,
- end-to-end output drift on a mixed depthwise/separable/Dense stack,
- average batch latency for the float and TurboQuant models.

Run:

```bash
python tensorflow/python/ops/turboquant/benchmark_turboquant.py \
  --json_output /tmp/turboquant_benchmark.json
```
