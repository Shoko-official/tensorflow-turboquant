"""Minimal benchmark for TurboQuant convolution and Dense models."""

import time

import numpy as np

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.ops.turboquant.api import quantize_model
from tensorflow.python.ops.turboquant.api import summarize_model
from tensorflow.python.ops.turboquant.config import TurboQuantConfig


def _measure_latency(model, inputs, warmup=10, steps=50):
  for _ in range(warmup):
    model(inputs)

  start = time.perf_counter()
  for _ in range(steps):
    model(inputs)
  return (time.perf_counter() - start) / float(steps)


def main():
  rng = np.random.default_rng(123)
  inputs = rng.normal(size=(32, 32, 32, 3)).astype(np.float32)

  model = Sequential([
      InputLayer(input_shape=(32, 32, 3)),
      Conv2D(16, 3, padding='same', activation='relu'),
      Conv2D(32, 3, strides=2, padding='same', activation='relu'),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(64),
  ])
  model(inputs)

  config = TurboQuantConfig(num_bits=4, group_size=64, outlier_threshold=6.0)
  quantized_model = quantize_model(model, config)

  reference_outputs = model(inputs).numpy()
  quantized_outputs = quantized_model(inputs).numpy()
  summaries = summarize_model(model, config)

  print('TurboQuant summaries:')
  total_original_bytes = 0.0
  total_packed_bytes = 0.0
  for summary in summaries:
    total_original_bytes += summary['original_bytes']
    total_packed_bytes += summary['packed_bytes']
    print(
        f"  - {summary['layer_name']} ({summary['layer_type']}): "
        f"ratio={summary['compression_ratio']:.2f}x, "
        f"mse={summary['mean_squared_error']:.6f}, "
        f"max_abs_error={summary['max_abs_error']:.6f}"
    )

  print(
      'Total effective compression: '
      f'{total_original_bytes / total_packed_bytes:.2f}x'
  )
  print(
      'Output drift: '
      f'mse={np.mean(np.square(reference_outputs - quantized_outputs)):.6f}, '
      f'max_abs={np.max(np.abs(reference_outputs - quantized_outputs)):.6f}'
  )
  print(
      'Average latency per batch: '
      f'float={_measure_latency(model, inputs) * 1e3:.3f} ms, '
      f'turboquant={_measure_latency(quantized_model, inputs) * 1e3:.3f} ms'
  )


if __name__ == '__main__':
  main()
