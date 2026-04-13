"""Profiling helper for TurboQuant quantization and inference stages."""

import argparse
import json
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


def _build_profile_model():
  return Sequential([
      InputLayer(input_shape=(28, 28, 8)),
      Conv2D(16, 3, padding='same', activation='relu'),
      Conv2D(16, 3, padding='same', activation='relu'),
      Flatten(),
      Dense(64, activation='relu'),
      Dense(10),
  ])


def _measure_average_ms(fn, steps):
  start = time.perf_counter()
  for _ in range(steps):
    fn()
  return ((time.perf_counter() - start) * 1e3) / float(steps)


def _measure_samples_ms(fn, steps):
  samples = []
  for _ in range(steps):
    start = time.perf_counter()
    fn()
    samples.append((time.perf_counter() - start) * 1e3)
  return samples


def _summarize_samples(samples):
  values = np.asarray(samples, dtype=np.float64)
  return {
      'count': int(values.size),
      'min': float(np.min(values)),
      'max': float(np.max(values)),
      'mean': float(np.mean(values)),
      'p50': float(np.percentile(values, 50)),
      'p95': float(np.percentile(values, 95)),
  }


def _build_hotspot_rank(distributions):
  p50_total = float(sum(stage['p50'] for stage in distributions.values()))
  p95_total = float(sum(stage['p95'] for stage in distributions.values()))
  hotspots = []
  for stage_name, stats in distributions.items():
    p50_share = float(stats['p50'] / p50_total) if p50_total > 0 else 0.0
    p95_share = float(stats['p95'] / p95_total) if p95_total > 0 else 0.0
    hotspots.append({
        'stage': stage_name,
        'p50_ms': float(stats['p50']),
        'p95_ms': float(stats['p95']),
        'p50_share': p50_share,
        'p95_share': p95_share,
    })
  hotspots.sort(key=lambda item: item['p95_share'], reverse=True)
  return hotspots


def _profile(seed, batch_size, warmup_steps, benchmark_steps, config):
  rng = np.random.default_rng(seed)
  inputs = rng.normal(size=(batch_size, 28, 28, 8)).astype(np.float32)
  model = _build_profile_model()
  model(inputs)

  summarize_start = time.perf_counter()
  summary_report = summarize_model(
      model, config, include_skipped=True, return_report=True
  )
  summarize_ms = (time.perf_counter() - summarize_start) * 1e3

  quantize_start = time.perf_counter()
  quantized_report = quantize_model(model, config, return_report=True)
  quantize_ms = (time.perf_counter() - quantize_start) * 1e3
  quantized_model = quantized_report['model']

  for _ in range(warmup_steps):
    model(inputs)
    quantized_model(inputs)

  float_inference_ms = _measure_average_ms(lambda: model(inputs), benchmark_steps)
  float_samples_ms = _measure_samples_ms(lambda: model(inputs), benchmark_steps)
  turbo_inference_ms = _measure_average_ms(
      lambda: quantized_model(inputs), benchmark_steps
  )
  turbo_samples_ms = _measure_samples_ms(
      lambda: quantized_model(inputs), benchmark_steps
  )

  reference = model(inputs).numpy()
  quantized = quantized_model(inputs).numpy()
  diff = reference - quantized

  timing_distributions = {
      'summarize_model': _summarize_samples([summarize_ms]),
      'quantize_model': _summarize_samples([quantize_ms]),
      'float_inference': _summarize_samples(float_samples_ms),
      'turboquant_inference': _summarize_samples(turbo_samples_ms),
  }
  hotspot_rank = _build_hotspot_rank(timing_distributions)

  return {
      'seed': int(seed),
      'batch_size': int(batch_size),
      'warmup_steps': int(warmup_steps),
      'benchmark_steps': int(benchmark_steps),
      'quantization_config': config.to_dict(),
      'timings_ms': {
          'summarize_model': float(summarize_ms),
          'quantize_model': float(quantize_ms),
          'float_inference_avg': float(float_inference_ms),
          'turboquant_inference_avg': float(turbo_inference_ms),
      },
      'timing_distributions_ms': timing_distributions,
      'hotspots': hotspot_rank,
      'aggregate': dict(quantized_report['aggregate']),
      'layer_summaries': quantized_report['summaries'],
      'drift': {
          'mean_squared_error': float(np.mean(np.square(diff))),
          'max_abs_error': float(np.max(np.abs(diff))),
      },
      'pre_quantization_aggregate': dict(summary_report['aggregate']),
  }


def _print_report(report):
  print('TurboQuant profile report')
  print(
      'Stage timings: '
      f"summarize={report['timings_ms']['summarize_model']:.2f} ms, "
      f"quantize={report['timings_ms']['quantize_model']:.2f} ms, "
      f"float_inference={report['timings_ms']['float_inference_avg']:.3f} ms, "
      f"turbo_inference={report['timings_ms']['turboquant_inference_avg']:.3f} ms"
  )
  print(
      'Compression: '
      f"{report['aggregate']['effective_compression_ratio']:.2f}x, "
      f"quantized_layers={report['aggregate']['quantized_layers']}, "
      f"skipped_layers={report['aggregate']['skipped_layers']}"
  )
  print(
      'Output drift: '
      f"mse={report['drift']['mean_squared_error']:.6f}, "
      f"max_abs={report['drift']['max_abs_error']:.6f}"
  )
  top_hotspots = report.get('hotspots', [])
  if top_hotspots:
    preview = top_hotspots[:3]
    formatted = ', '.join(
        f"{item['stage']} (p95_share={item['p95_share']:.1%})"
        for item in preview
    )
    print(f'Hotspots (p95 share): {formatted}')


def main():
  parser = argparse.ArgumentParser(description='Profile TurboQuant stages.')
  parser.add_argument('--seed', type=int, default=123)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--warmup_steps', type=int, default=10)
  parser.add_argument('--benchmark_steps', type=int, default=50)
  parser.add_argument('--num_bits', type=int, default=4)
  parser.add_argument('--group_size', type=int, default=8)
  parser.add_argument('--outlier_threshold', type=float, default=6.0)
  parser.add_argument(
      '--json_output',
      type=str,
      default='',
      help='Optional path to write JSON profiling report.',
  )
  args = parser.parse_args()

  config = TurboQuantConfig(
      num_bits=args.num_bits,
      group_size=args.group_size,
      outlier_threshold=args.outlier_threshold,
  )
  report = _profile(
      seed=args.seed,
      batch_size=args.batch_size,
      warmup_steps=args.warmup_steps,
      benchmark_steps=args.benchmark_steps,
      config=config,
  )
  _print_report(report)
  if args.json_output:
    with open(args.json_output, 'w', encoding='utf-8') as output_file:
      json.dump(report, output_file, indent=2, sort_keys=True)


if __name__ == '__main__':
  main()
