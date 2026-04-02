"""Runs TurboQuant ablations across quantization hyper-parameters."""

import argparse
import json
import itertools

import numpy as np

from tensorflow.python.ops.turboquant.benchmark_turboquant import _run_suite
from tensorflow.python.ops.turboquant.config import TurboQuantConfig


def _parse_csv_ints(value):
  return tuple(int(item.strip()) for item in value.split(',') if item.strip())


def _parse_csv_floats(value):
  return tuple(float(item.strip()) for item in value.split(',') if item.strip())


def _parse_csv_strings(value):
  return tuple(item.strip() for item in value.split(',') if item.strip())


def run_ablations(
    num_bits_values,
    group_size_values,
    outlier_threshold_values,
    seeds,
    models,
    batch_sizes,
    warmup_steps,
    benchmark_steps,
    repeats,
    objective_lambda_mse=1.0,
):
  """Evaluates candidate quantization settings and ranks configurations."""
  runs = []
  for num_bits, group_size, outlier_threshold in itertools.product(
      num_bits_values, group_size_values, outlier_threshold_values
  ):
    per_seed = []
    for seed in seeds:
      config = TurboQuantConfig(
          num_bits=num_bits,
          group_size=group_size,
          outlier_threshold=outlier_threshold,
      )
      report = _run_suite(
          config=config,
          models=models,
          batch_sizes=batch_sizes,
          seed=seed,
          warmup=warmup_steps,
          steps=benchmark_steps,
          repeats=repeats,
      )
      per_seed.append(report['summary'])

    compression_values = [
        float(item['compression_ratio']['mean']) for item in per_seed
    ]
    mse_values = [float(item['output_mse']['mean']) for item in per_seed]
    speedup_values = [
        float(item['latency_speedup_p50_float_over_turboquant']['mean'])
        for item in per_seed
    ]
    objective = (
        float(np.mean(compression_values))
        + float(np.mean(speedup_values))
        - (objective_lambda_mse * float(np.mean(mse_values)))
    )
    runs.append({
        'config': {
            'num_bits': int(num_bits),
            'group_size': int(group_size),
            'outlier_threshold': float(outlier_threshold),
        },
        'seed_count': int(len(seeds)),
        'compression_ratio_mean': float(np.mean(compression_values)),
        'output_mse_mean': float(np.mean(mse_values)),
        'speedup_mean': float(np.mean(speedup_values)),
        'objective': float(objective),
    })

  runs.sort(key=lambda item: item['objective'], reverse=True)
  return {
      'metadata': {
          'models': list(models),
          'batch_sizes': [int(item) for item in batch_sizes],
          'seeds': [int(item) for item in seeds],
          'objective_lambda_mse': float(objective_lambda_mse),
      },
      'runs': runs,
      'best': runs[0] if runs else None,
  }


def _print_report(report):
  best = report['best']
  print('TurboQuant ablation search')
  print(
      f"Runs: {len(report['runs'])} | Seeds: {len(report['metadata']['seeds'])}"
  )
  if not best:
    return
  print(
      'Best config: '
      f"num_bits={best['config']['num_bits']}, "
      f"group_size={best['config']['group_size']}, "
      f"outlier_threshold={best['config']['outlier_threshold']}"
  )
  print(
      'Best metrics: '
      f"compression={best['compression_ratio_mean']:.3f}, "
      f"mse={best['output_mse_mean']:.6f}, "
      f"speedup={best['speedup_mean']:.3f}, "
      f"objective={best['objective']:.6f}"
  )


def main():
  parser = argparse.ArgumentParser(description='Run TurboQuant ablation search.')
  parser.add_argument('--num_bits', type=str, default='2,3,4')
  parser.add_argument('--group_sizes', type=str, default='8,16,32')
  parser.add_argument('--outlier_thresholds', type=str, default='4.0,6.0,8.0')
  parser.add_argument('--seeds', type=str, default='123,456,789')
  parser.add_argument(
      '--models',
      type=str,
      default='dense_mlp,conv2d_stack,depthwise_separable',
  )
  parser.add_argument('--batch_sizes', type=str, default='1,8')
  parser.add_argument('--warmup_steps', type=int, default=5)
  parser.add_argument('--benchmark_steps', type=int, default=20)
  parser.add_argument('--repeats', type=int, default=2)
  parser.add_argument('--objective_lambda_mse', type=float, default=1.0)
  parser.add_argument('--json_output', type=str, default='')
  args = parser.parse_args()

  report = run_ablations(
      num_bits_values=_parse_csv_ints(args.num_bits),
      group_size_values=_parse_csv_ints(args.group_sizes),
      outlier_threshold_values=_parse_csv_floats(args.outlier_thresholds),
      seeds=_parse_csv_ints(args.seeds),
      models=_parse_csv_strings(args.models),
      batch_sizes=_parse_csv_ints(args.batch_sizes),
      warmup_steps=args.warmup_steps,
      benchmark_steps=args.benchmark_steps,
      repeats=args.repeats,
      objective_lambda_mse=args.objective_lambda_mse,
  )
  _print_report(report)
  if args.json_output:
    with open(args.json_output, 'w', encoding='utf-8') as output_file:
      json.dump(report, output_file, indent=2, sort_keys=True)


if __name__ == '__main__':
  main()
