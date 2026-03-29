"""Tests for TurboQuant core routines."""

import unittest

import numpy as np

from tensorflow.python.ops.turboquant.config import TurboQuantConfig
from tensorflow.python.ops.turboquant.config import CalibrationConfig
from tensorflow.python.ops.turboquant.core import dequantize_tensor
from tensorflow.python.ops.turboquant.core import estimate_packed_bytes
from tensorflow.python.ops.turboquant.core import original_bytes
from tensorflow.python.ops.turboquant.core import quantize_tensor
from tensorflow.python.ops.turboquant.core import summarize_encoding


class TurboQuantCoreTest(unittest.TestCase):

  def test_round_trip_preserves_shape_and_limits_error(self):
    rng = np.random.default_rng(1234)
    tensor = rng.normal(size=(96, 48)).astype(np.float32)
    config = TurboQuantConfig(num_bits=4, group_size=32)

    encoding = quantize_tensor(tensor, config)
    restored = dequantize_tensor(encoding)

    self.assertEqual(restored.shape, tensor.shape)
    self.assertLess(np.mean(np.square(tensor - restored)), 0.02)
    self.assertLess(np.max(np.abs(tensor - restored)), 0.5)

  def test_outlier_residual_reduces_peak_error(self):
    tensor = np.zeros((128, 16), dtype=np.float32)
    tensor[:, :] = np.linspace(-0.25, 0.25, num=tensor.size).reshape(tensor.shape)
    tensor[17, 3] = 9.0

    base_config = TurboQuantConfig(
        num_bits=4, group_size=32, outlier_threshold=0.0
    )
    robust_config = TurboQuantConfig(
        num_bits=4, group_size=32, outlier_threshold=4.0
    )

    base_error = np.max(
        np.abs(tensor - dequantize_tensor(quantize_tensor(tensor, base_config)))
    )
    robust_error = np.max(
        np.abs(tensor - dequantize_tensor(quantize_tensor(tensor, robust_config)))
    )

    self.assertLess(robust_error, base_error)

  def test_estimated_packed_bytes_is_smaller_than_float32_tensor(self):
    rng = np.random.default_rng(7)
    tensor = rng.normal(size=(256, 64)).astype(np.float32)

    encoding = quantize_tensor(tensor, TurboQuantConfig())

    self.assertLess(estimate_packed_bytes(encoding), original_bytes(tensor))

  def test_summary_contains_expected_metrics(self):
    rng = np.random.default_rng(21)
    tensor = rng.normal(size=(64, 12)).astype(np.float32)
    encoding = quantize_tensor(tensor, TurboQuantConfig(group_size=16))

    summary = summarize_encoding(tensor, encoding)

    self.assertIn('compression_ratio', summary)
    self.assertIn('mean_squared_error', summary)
    self.assertGreater(summary['compression_ratio'], 1.0)

  def test_quantize_rejects_non_finite_values(self):
    tensor = np.ones((8, 8), dtype=np.float32)
    tensor[3, 2] = np.nan
    with self.assertRaisesRegex(ValueError, 'finite'):
      quantize_tensor(tensor, TurboQuantConfig())

  def test_encoding_uses_compact_index_dtype(self):
    tensor = np.linspace(-1.0, 1.0, num=1024, dtype=np.float32).reshape(64, 16)
    encoding = quantize_tensor(
        tensor, TurboQuantConfig(num_bits=4, group_size=8)
    )
    self.assertEqual(encoding.indices.dtype, np.uint8)

  def test_round_trip_supports_non_terminal_axis(self):
    rng = np.random.default_rng(123)
    tensor = rng.normal(size=(5, 7, 11)).astype(np.float32)
    config = TurboQuantConfig(num_bits=4, group_size=8, axis=1)

    encoding = quantize_tensor(tensor, config)
    restored = dequantize_tensor(encoding)

    self.assertEqual(restored.shape, tensor.shape)
    self.assertLess(np.mean(np.square(tensor - restored)), 0.06)

  def test_config_from_dict_rejects_unknown_keys(self):
    with self.assertRaisesRegex(ValueError, 'Unknown `TurboQuantConfig` keys'):
      TurboQuantConfig.from_dict({'group_size': 8, 'unknown_key': 1})
    with self.assertRaisesRegex(ValueError, 'Unknown `CalibrationConfig` keys'):
      CalibrationConfig.from_dict({'max_steps': 8, 'bad': 1})


if __name__ == '__main__':
  unittest.main()
