"""Scientific aggregation utilities for TurboQuant benchmark outputs."""

import argparse
import json
from pathlib import Path

import numpy as np


def _bootstrap_ci(values, samples=2000, alpha=0.05, seed=123):
  values = np.asarray(values, dtype=np.float64)
  if values.size == 0:
    return {'low': np.nan, 'high': np.nan}
  if values.size == 1:
    only = float(values[0])
    return {'low': only, 'high': only}

  rng = np.random.default_rng(seed)
  means = np.zeros((samples,), dtype=np.float64)
  for index in range(samples):
    bootstrap_sample = rng.choice(values, size=values.size, replace=True)
    means[index] = np.mean(bootstrap_sample)
  low = float(np.percentile(means, 100.0 * (alpha / 2.0)))
  high = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
  return {'low': low, 'high': high}


def _extract_metrics(report):
  metrics = []
  if 'cases' in report:
    for case in report['cases']:
      metrics.append({
          'compression_ratio': float(case['compression']['effective_ratio']),
          'mse': float(case['output_drift']['mean_squared_error']),
          'agreement': np.nan,
          'latency_p50_float': float(case['latency']['float_model']['latency_ms']['p50']),
          'latency_p50_turbo': float(case['latency']['turboquant_model']['latency_ms']['p50']),
      })
    return metrics

  if 'results' in report:
    for result in report['results']:
      metrics.append({
          'compression_ratio': float(
              result['compression']['turboquant']['effective_compression_ratio']
          ),
          'mse': float(result['drift']['turboquant_mse']),
          'agreement': float(result['drift']['argmax_agreement_turboquant']),
          'latency_p50_float': float(result['latency_ms']['float']['p50']),
          'latency_p50_turbo': float(result['latency_ms']['turboquant']['p50']),
      })
    return metrics

  raise ValueError('Unsupported benchmark report schema.')


def _summarize_metric(values):
  values = np.asarray(values, dtype=np.float64)
  valid = values[np.isfinite(values)]
  if valid.size == 0:
    return {
        'count': 0,
        'mean': np.nan,
        'std': np.nan,
        'ci95': {'low': np.nan, 'high': np.nan},
    }
  return {
      'count': int(valid.size),
      'mean': float(np.mean(valid)),
      'std': float(np.std(valid)),
      'ci95': _bootstrap_ci(valid, samples=2000, alpha=0.05),
  }


def analyze_reports(report_paths):
  """Aggregates metrics across one or many TurboQuant benchmark reports."""
  rows = []
  for report_path in report_paths:
    with open(report_path, 'r', encoding='utf-8') as input_file:
      report = json.load(input_file)
    rows.extend(_extract_metrics(report))

  compression = [row['compression_ratio'] for row in rows]
  mse = [row['mse'] for row in rows]
  agreement = [row['agreement'] for row in rows]
  speedup = [
      row['latency_p50_float'] / max(row['latency_p50_turbo'], 1e-9) for row in rows
  ]
  return {
      'report_count': int(len(report_paths)),
      'case_count': int(len(rows)),
      'compression_ratio': _summarize_metric(compression),
      'output_mse': _summarize_metric(mse),
      'argmax_agreement': _summarize_metric(agreement),
      'p50_speedup_float_over_turboquant': _summarize_metric(speedup),
  }


def _print_summary(summary):
  print('TurboQuant statistical summary')
  print(f"Reports: {summary['report_count']} | Cases: {summary['case_count']}")
  for key in (
      'compression_ratio',
      'output_mse',
      'argmax_agreement',
      'p50_speedup_float_over_turboquant',
  ):
    item = summary[key]
    print(
        f"  - {key}: mean={item['mean']:.6f}, std={item['std']:.6f}, "
        f"ci95=[{item['ci95']['low']:.6f}, {item['ci95']['high']:.6f}]"
    )


def main():
  parser = argparse.ArgumentParser(
      description='Aggregate TurboQuant benchmark reports with confidence intervals.'
  )
  parser.add_argument(
      'reports',
      nargs='+',
      help='JSON report files produced by benchmark scripts.',
  )
  parser.add_argument('--json_output', type=str, default='')
  args = parser.parse_args()

  report_paths = [str(Path(item)) for item in args.reports]
  summary = analyze_reports(report_paths)
  _print_summary(summary)
  if args.json_output:
    with open(args.json_output, 'w', encoding='utf-8') as output_file:
      json.dump(summary, output_file, indent=2, sort_keys=True)


if __name__ == '__main__':
  main()
