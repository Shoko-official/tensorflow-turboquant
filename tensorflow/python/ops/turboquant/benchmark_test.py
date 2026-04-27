"""Tests for TurboQuant benchmark and profiling tools."""

import json
import os

from tensorflow.python.ops.turboquant.analyze_turboquant_results import analyze_reports
from tensorflow.python.ops.turboquant.benchmark_turboquant import _run_suite
from tensorflow.python.ops.turboquant.benchmark_real_models import validate_quality_gates
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
    self.assertIn('timing_distributions_ms', report)
    self.assertIn('hotspots', report)
    self.assertIn('aggregate', report)
    self.assertIn('drift', report)
    self.assertGreaterEqual(len(report['hotspots']), 1)
    top_hotspot = report['hotspots'][0]
    self.assertIn('stage', top_hotspot)
    self.assertIn('p50_share', top_hotspot)
    self.assertIn('p95_share', top_hotspot)
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
    self.assertIn('compression_ratio', report['summary'])
    self.assertIn('argmax_agreement', report['summary'])
    self.assertIn('accuracy_delta', report['summary'])
    self.assertIn('confidence_interval_95',
                  report['summary']['argmax_agreement']['turboquant'])

  def test_real_model_quality_gate_validation(self):
    report = {
        'summary': {
            'accuracy_delta': {
                'turboquant': {'min': -0.03},
                'baseline': {'min': -0.01},
            },
            'argmax_agreement': {
                'turboquant': {'min': 0.96},
                'baseline': {'min': 0.98},
            },
            'output_mse': {
                'turboquant': {'max': 0.02},
                'baseline': {'max': 0.01},
            },
        }
    }
    self.assertEqual(
        [],
        validate_quality_gates(
            report,
            max_turboquant_accuracy_drop=0.05,
            max_baseline_accuracy_drop=0.05,
            min_turboquant_argmax_agreement=0.95,
            min_baseline_argmax_agreement=0.95,
            max_turboquant_mse=0.05,
        ),
    )
    failures = validate_quality_gates(
        report,
        max_turboquant_accuracy_drop=0.02,
        min_turboquant_argmax_agreement=0.97,
        max_turboquant_mse=0.01,
    )
    self.assertLen(failures, 3)
    self.assertTrue(
        any('accuracy drop gate failed' in failure for failure in failures)
    )
    self.assertTrue(
        any('argmax agreement gate failed' in failure for failure in failures)
    )
    self.assertTrue(any('output MSE gate failed' in failure for failure in failures))

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
