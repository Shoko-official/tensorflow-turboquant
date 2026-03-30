"""Tests for TurboQuant benchmark and profiling tools."""

from tensorflow.python.ops.turboquant.benchmark_turboquant import _run_suite
from tensorflow.python.ops.turboquant.config import TurboQuantConfig
from tensorflow.python.ops.turboquant.profile_turboquant import _profile
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


if __name__ == '__main__':
  test.main()
