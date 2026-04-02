"""Tests for TurboQuant benchmark and profiling tools."""

import json
import os

from tensorflow.python.ops.turboquant.analyze_turboquant_results import analyze_reports
from tensorflow.python.ops.turboquant.benchmark_turboquant import _run_suite
from tensorflow.python.ops.turboquant.benchmark_real_models import run_real_model_benchmark
from tensorflow.python.ops.turboquant.config import TurboQuantConfig
from tensorflow.python.ops.turboquant.profile_turboquant import _profile
from tensorflow.python.ops.turboquant.run_turboquant_ablations import run_ablations
from tensorflow.python.platform import test


class TurboBenchmarkToolsTest(test.TestCase):

  def test_benchmark_suite_returns_expected_sections(self):
    config = TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=6.0)
    report = _run_suite(
        config=config,
        models=('dense_mlp',),
        batch_sizes=(1,),
        seed=123,
        warmup=0,
        steps=1,
        repeats=1,
    )

    self.assertIn('metadata', report)
    self.assertIn('summary', report)
    self.assertIn('cases', report)
    self.assertLen(report['cases'], 1)
    case = report['cases'][0]
    self.assertIn('compression', case)
    self.assertIn('latency', case)
    self.assertIn('output_drift', case)

  def test_profile_returns_stage_timings_and_aggregate(self):
    config = TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=6.0)
    report = _profile(
        seed=123,
        batch_size=2,
        warmup_steps=0,
        benchmark_steps=1,
        config=config,
    )

    self.assertIn('timings_ms', report)
    self.assertIn('aggregate', report)
    self.assertIn('drift', report)
    self.assertGreaterEqual(report['aggregate']['supported_layers'], 1)

  def test_real_model_benchmark_runs_with_synthetic_data(self):
    config = TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=6.0)
    baseline = TurboQuantConfig(num_bits=4, group_size=64, outlier_threshold=0.0)
    report = run_real_model_benchmark(
        models=('separable_cnn',),
        seeds=(123,),
        config=config,
        baseline_config=baseline,
        dataset_source='synthetic',
        train_epochs=0,
        sample_count=32,
        eval_count=16,
        batch_size=8,
        warmup_steps=0,
        benchmark_steps=1,
        num_classes=5,
    )
    self.assertEqual(report['summary']['case_count'], 1)
    self.assertIn('turboquant_effective_compression_ratio_mean', report['summary'])

  def test_analyze_reports_aggregates_generated_report(self):
    config = TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=6.0)
    report = _run_suite(
        config=config,
        models=('dense_mlp',),
        batch_sizes=(1,),
        seed=123,
        warmup=0,
        steps=1,
        repeats=1,
    )
    report_path = os.path.join(self.get_temp_dir(), 'turboquant_suite.json')
    with open(report_path, 'w', encoding='utf-8') as output_file:
      json.dump(report, output_file)
    summary = analyze_reports([report_path])
    self.assertEqual(summary['report_count'], 1)
    self.assertGreaterEqual(summary['case_count'], 1)

  def test_ablation_runner_returns_ranked_runs(self):
    report = run_ablations(
        num_bits_values=(4,),
        group_size_values=(8, 16),
        outlier_threshold_values=(4.0,),
        seeds=(123,),
        models=('dense_mlp',),
        batch_sizes=(1,),
        warmup_steps=0,
        benchmark_steps=1,
        repeats=1,
        objective_lambda_mse=1.0,
    )
    self.assertLen(report['runs'], 2)
    self.assertIsNotNone(report['best'])


if __name__ == '__main__':
  test.main()
