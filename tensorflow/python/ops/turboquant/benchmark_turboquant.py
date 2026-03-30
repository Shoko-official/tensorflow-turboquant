"""Reproducible TurboQuant benchmark suite."""

import argparse
import json
import platform
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


def _parse_csv_ints(value):
  return tuple(int(item.strip()) for item in value.split(',') if item.strip())


def _parse_csv_strings(value):
  return tuple(item.strip() for item in value.split(',') if item.strip())


def _measure_step_latencies_ms(model, inputs, warmup, steps, repeats):
  for _ in range(warmup):
    model(inputs)

  latencies_ms = []
  for _ in range(repeats):
    for _ in range(steps):
      start = time.perf_counter()
      model(inputs)
      latencies_ms.append((time.perf_counter() - start) * 1e3)
  return np.asarray(latencies_ms, dtype=np.float64)


def _latency_summary(latencies_ms, batch_size):
  throughput = (batch_size * 1000.0) / np.maximum(latencies_ms, 1e-9)
  return {
      'samples': int(latencies_ms.size),
      'latency_ms': {
          'mean': float(np.mean(latencies_ms)),
          'p50': float(np.percentile(latencies_ms, 50)),
          'p95': float(np.percentile(latencies_ms, 95)),
      },
      'throughput_items_per_s': {
          'mean': float(np.mean(throughput)),
          'p50': float(np.percentile(throughput, 50)),
          'p95': float(np.percentile(throughput, 95)),
      },
  }


def _build_dense_mlp():
  return Sequential([
      InputLayer(input_shape=(1024,)),
      Dense(1024, activation='relu'),
      Dense(768, activation='relu'),
      Dense(256),
  ])


def _build_conv2d_stack():
  return Sequential([
      InputLayer(input_shape=(32, 32, 16)),
      Conv2D(32, 3, padding='same', activation='relu'),
      Conv2D(32, 3, padding='same', activation='relu'),
      Conv2D(16, 1),
      Flatten(),
      Dense(128),
  ])


def _build_depthwise_separable_stack():
  return Sequential([
      InputLayer(input_shape=(24, 24, 64)),
      DepthwiseConv2D(5, padding='same', depth_multiplier=1, activation='relu'),
      SeparableConv2D(32, 5, padding='same', depth_multiplier=1, activation='relu'),
      Conv2D(32, 3, strides=2, padding='same', activation='relu'),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(64),
  ])


_MODEL_BUILDERS = {
    'dense_mlp': (_build_dense_mlp, (1024,)),
    'conv2d_stack': (_build_conv2d_stack, (32, 32, 16)),
    'depthwise_separable': (_build_depthwise_separable_stack, (24, 24, 64)),
}


def _run_case(model_name, batch_size, config, seed, warmup, steps, repeats):
  builder, input_shape = _MODEL_BUILDERS[model_name]
  rng = np.random.default_rng(seed + (batch_size * 997))
  inputs = rng.normal(size=(batch_size,) + input_shape).astype(np.float32)

  model = builder()
  model(inputs)

  quantize_start = time.perf_counter()
  report = quantize_model(model, config, return_report=True)
  quantize_ms = (time.perf_counter() - quantize_start) * 1e3
  quantized_model = report['model']
  aggregate = report['aggregate']

  reference_outputs = model(inputs).numpy()
  quantized_outputs = quantized_model(inputs).numpy()
  drift = reference_outputs - quantized_outputs

  float_latencies = _measure_step_latencies_ms(
      model, inputs, warmup=warmup, steps=steps, repeats=repeats
  )
  turbo_latencies = _measure_step_latencies_ms(
      quantized_model, inputs, warmup=warmup, steps=steps, repeats=repeats
  )

  return {
      'model_name': model_name,
      'batch_size': int(batch_size),
      'quantization_config': config.to_dict(),
      'quantization_wall_time_ms': float(quantize_ms),
      'compression': {
          'original_bytes': float(aggregate['total_original_bytes']),
          'packed_bytes': float(aggregate['total_packed_bytes']),
          'effective_ratio': float(aggregate['effective_compression_ratio']),
      },
      'layer_coverage': {
          'supported_layers': int(aggregate['supported_layers']),
          'quantized_layers': int(aggregate['quantized_layers']),
          'skipped_layers': int(aggregate['skipped_layers']),
          'skipped_reasons': dict(aggregate['skipped_reasons']),
      },
      'output_drift': {
          'mean_squared_error': float(np.mean(np.square(drift))),
          'max_abs_error': float(np.max(np.abs(drift))),
      },
      'latency': {
          'float_model': _latency_summary(float_latencies, batch_size),
          'turboquant_model': _latency_summary(turbo_latencies, batch_size),
      },
  }


def _run_suite(config, models, batch_sizes, seed, warmup, steps, repeats):
  cases = []
  for model_name in models:
    for batch_size in batch_sizes:
      case_seed = seed + (len(cases) * 131)
      cases.append(
          _run_case(
              model_name=model_name,
              batch_size=batch_size,
              config=config,
              seed=case_seed,
              warmup=warmup,
              steps=steps,
              repeats=repeats,
          )
      )

  ratios = [case['compression']['effective_ratio'] for case in cases]
  mse_values = [case['output_drift']['mean_squared_error'] for case in cases]
  speedup_values = []
  for case in cases:
    float_p50 = case['latency']['float_model']['latency_ms']['p50']
    turbo_p50 = case['latency']['turboquant_model']['latency_ms']['p50']
    speedup_values.append(float_p50 / max(turbo_p50, 1e-9))

  return {
      'metadata': {
          'benchmark_name': 'turboquant_suite',
          'seed': int(seed),
          'model_count': int(len(models)),
          'batch_size_count': int(len(batch_sizes)),
          'cases': int(len(cases)),
          'steps_per_repeat': int(steps),
          'repeats': int(repeats),
          'warmup_steps': int(warmup),
          'platform': {
              'python': platform.python_version(),
              'system': platform.system(),
              'release': platform.release(),
              'processor': platform.processor(),
          },
      },
      'summary': {
          'compression_ratio': {
              'mean': float(np.mean(ratios)),
              'min': float(np.min(ratios)),
              'max': float(np.max(ratios)),
          },
          'output_mse': {
              'mean': float(np.mean(mse_values)),
              'max': float(np.max(mse_values)),
          },
          'latency_speedup_p50_float_over_turboquant': {
              'mean': float(np.mean(speedup_values)),
              'min': float(np.min(speedup_values)),
              'max': float(np.max(speedup_values)),
          },
      },
      'cases': cases,
  }


def _print_suite_report(report):
  print('TurboQuant benchmark suite')
  print(
      'Cases: '
      f"{report['metadata']['cases']} "
      f"(models={report['metadata']['model_count']}, "
      f"batch_sizes={report['metadata']['batch_size_count']})"
  )
  print(
      'Compression ratio (effective): '
      f"mean={report['summary']['compression_ratio']['mean']:.2f}x, "
      f"min={report['summary']['compression_ratio']['min']:.2f}x, "
      f"max={report['summary']['compression_ratio']['max']:.2f}x"
  )
  print(
      'Output drift MSE: '
      f"mean={report['summary']['output_mse']['mean']:.6f}, "
      f"max={report['summary']['output_mse']['max']:.6f}"
  )
  print(
      'Latency speedup p50 (float/turboquant): '
      f"mean={report['summary']['latency_speedup_p50_float_over_turboquant']['mean']:.2f}x, "
      f"min={report['summary']['latency_speedup_p50_float_over_turboquant']['min']:.2f}x, "
      f"max={report['summary']['latency_speedup_p50_float_over_turboquant']['max']:.2f}x"
  )
  print('')
  print('Per-case details:')
  for case in report['cases']:
    print(
        f"  - {case['model_name']} batch={case['batch_size']}: "
        f"ratio={case['compression']['effective_ratio']:.2f}x, "
        f"mse={case['output_drift']['mean_squared_error']:.6f}, "
        f"float_p50={case['latency']['float_model']['latency_ms']['p50']:.3f}ms, "
        f"turbo_p50={case['latency']['turboquant_model']['latency_ms']['p50']:.3f}ms"
    )


def main():
  parser = argparse.ArgumentParser(description='Run the TurboQuant benchmark suite.')
  parser.add_argument('--seed', type=int, default=123)
  parser.add_argument('--warmup_steps', type=int, default=10)
  parser.add_argument('--benchmark_steps', type=int, default=50)
  parser.add_argument('--repeats', type=int, default=3)
  parser.add_argument(
      '--batch_sizes',
      type=str,
      default='1,8,32',
      help='Comma-separated batch sizes.',
  )
  parser.add_argument(
      '--models',
      type=str,
      default='dense_mlp,conv2d_stack,depthwise_separable',
      help='Comma-separated model names.',
  )
  parser.add_argument('--num_bits', type=int, default=4)
  parser.add_argument('--group_size', type=int, default=8)
  parser.add_argument('--outlier_threshold', type=float, default=6.0)
  parser.add_argument(
      '--json_output',
      type=str,
      default='',
      help='Optional path to write the benchmark JSON report.',
  )
  args = parser.parse_args()

  batch_sizes = _parse_csv_ints(args.batch_sizes)
  model_names = _parse_csv_strings(args.models)
  unknown_models = [name for name in model_names if name not in _MODEL_BUILDERS]
  if unknown_models:
    raise ValueError(
        'Unknown model names: '
        + ', '.join(unknown_models)
        + '. Available: '
        + ', '.join(sorted(_MODEL_BUILDERS.keys()))
    )
  if not batch_sizes:
    raise ValueError('`--batch_sizes` must include at least one value.')
  if args.repeats < 1:
    raise ValueError('`--repeats` must be >= 1.')
  if args.benchmark_steps < 1:
    raise ValueError('`--benchmark_steps` must be >= 1.')
  if args.warmup_steps < 0:
    raise ValueError('`--warmup_steps` must be >= 0.')

  config = TurboQuantConfig(
      num_bits=args.num_bits,
      group_size=args.group_size,
      outlier_threshold=args.outlier_threshold,
  )
  report = _run_suite(
      config=config,
      models=model_names,
      batch_sizes=batch_sizes,
      seed=args.seed,
      warmup=args.warmup_steps,
      steps=args.benchmark_steps,
      repeats=args.repeats,
  )
  _print_suite_report(report)

  if args.json_output:
    with open(args.json_output, 'w', encoding='utf-8') as output_file:
      json.dump(report, output_file, indent=2, sort_keys=True)


if __name__ == '__main__':
  main()
