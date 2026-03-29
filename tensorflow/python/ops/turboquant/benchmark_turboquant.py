"""Minimal benchmark for TurboQuant convolution and Dense models."""

import argparse
import json
import time

import numpy as np

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import DepthwiseConv2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.layers import SeparableConv2D
from tensorflow.python.ops.turboquant.api import quantize_model
from tensorflow.python.ops.turboquant.config import TurboQuantConfig


def _measure_latency(model, inputs, warmup=10, steps=50):
  for _ in range(warmup):
    model(inputs)

  start = time.perf_counter()
  for _ in range(steps):
    model(inputs)
  return (time.perf_counter() - start) / float(steps)


def _build_reference_model():
  return Sequential([
      InputLayer(input_shape=(24, 24, 64)),
      DepthwiseConv2D(5, padding='same', depth_multiplier=1, activation='relu'),
      SeparableConv2D(32, 5, padding='same', depth_multiplier=1, activation='relu'),
      Conv2D(32, 3, strides=2, padding='same', activation='relu'),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(64),
  ])


def _run_benchmark(seed=123, batch_size=16, warmup=10, steps=50):
  rng = np.random.default_rng(seed)
  inputs = rng.normal(size=(batch_size, 24, 24, 64)).astype(np.float32)
  model = _build_reference_model()
  model(inputs)

  config = TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=6.0)
  report = quantize_model(model, config, return_report=True)
  quantized_model = report['model']
  summaries = report['summaries']
  aggregate = report['aggregate']

  reference_outputs = model(inputs).numpy()
  quantized_outputs = quantized_model(inputs).numpy()
  float_ms = _measure_latency(model, inputs, warmup=warmup, steps=steps) * 1e3
  turbo_ms = (
      _measure_latency(quantized_model, inputs, warmup=warmup, steps=steps) * 1e3
  )
  return {
      'seed': int(seed),
      'batch_size': int(batch_size),
      'warmup_steps': int(warmup),
      'benchmark_steps': int(steps),
      'quantization_config': config.to_dict(),
      'compression': {
          'original_bytes': float(aggregate['total_original_bytes']),
          'packed_bytes': float(aggregate['total_packed_bytes']),
          'ratio': float(aggregate['effective_compression_ratio']),
      },
      'aggregate': aggregate,
      'output_drift': {
          'mean_squared_error': float(
              np.mean(np.square(reference_outputs - quantized_outputs))
          ),
          'max_abs_error': float(
              np.max(np.abs(reference_outputs - quantized_outputs))
          ),
      },
      'latency_ms': {
          'float_model': float(float_ms),
          'turboquant_model': float(turbo_ms),
      },
      'layer_summaries': summaries,
  }


def _print_report(report):
  print('TurboQuant summaries:')
  for summary in report['layer_summaries']:
    print(
        f"  - {summary['layer_name']} ({summary['layer_type']}): "
        f"ratio={summary['compression_ratio']:.2f}x, "
        f"mse={summary['mean_squared_error']:.6f}, "
        f"max_abs_error={summary['max_abs_error']:.6f}"
    )

  print(
      'Total effective compression: '
      f"{report['compression']['ratio']:.2f}x"
  )
  print(
      'Output drift: '
      f"mse={report['output_drift']['mean_squared_error']:.6f}, "
      f"max_abs={report['output_drift']['max_abs_error']:.6f}"
  )
  print(
      'Average latency per batch: '
      f"float={report['latency_ms']['float_model']:.3f} ms, "
      f"turboquant={report['latency_ms']['turboquant_model']:.3f} ms"
  )


def main():
  parser = argparse.ArgumentParser(description='Run TurboQuant benchmark.')
  parser.add_argument('--seed', type=int, default=123)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--warmup_steps', type=int, default=10)
  parser.add_argument('--benchmark_steps', type=int, default=50)
  parser.add_argument(
      '--json_output',
      type=str,
      default='',
      help='Optional path to write JSON benchmark report.',
  )
  args = parser.parse_args()

  report = _run_benchmark(
      seed=args.seed,
      batch_size=args.batch_size,
      warmup=args.warmup_steps,
      steps=args.benchmark_steps,
  )
  _print_report(report)

  if args.json_output:
    with open(args.json_output, 'w', encoding='utf-8') as output_file:
      json.dump(report, output_file, indent=2, sort_keys=True)


if __name__ == '__main__':
  main()
