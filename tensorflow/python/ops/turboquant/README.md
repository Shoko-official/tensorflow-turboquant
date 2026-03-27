# TurboQuant

TurboQuant is a lightweight weight-only quantization path for TensorFlow Python
layers built around three conservative ideas:

- per-output-channel codebooks instead of one global codebook,
- block-wise scales to preserve local dynamic range,
- optional outlier residuals kept in full precision for stability.

The current integration targets `tf.keras.layers.Dense`,
`tf.keras.layers.Conv1D`, `tf.keras.layers.Conv2D`, and
`tf.keras.layers.Conv3D` through `TurboDense`, `TurboConv1D`,
`TurboConv2D`, `TurboConv3D`, and a `quantize_model()` cloning helper. The
core packing logic is kept in a NumPy-only module so the quantizer is easy to
test and extend before moving deeper into TensorFlow compiler paths.

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
- `export_saved_model()` emits a TensorFlow SavedModel with a stable
  `serving_default` signature for inference, and `load_saved_model()` restores
  the exported inference object through the core SavedModel loader.
- The tensor packing API is shape-generic, so extending support beyond the
  current `Dense` and convolution wrappers does not require reworking the
  quantizer itself.

## Benchmark

A minimal reproducible benchmark is available in
`tensorflow/python/ops/turboquant/benchmark_turboquant.py`. It reports:

- per-layer effective compression,
- end-to-end output drift on a small Dense stack,
- average batch latency for the float and TurboQuant models.
