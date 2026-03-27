"""Tests for TurboQuant calibration utilities."""

import numpy as np

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.ops.turboquant.api import summarize_model
from tensorflow.python.ops.turboquant.calibration import collect_calibration_stats
from tensorflow.python.ops.turboquant.config import CalibrationConfig
from tensorflow.python.ops.turboquant.config import TurboQuantConfig
from tensorflow.python.platform import test


class TurboCalibrationTest(test.TestCase):

  def test_collect_calibration_stats_reports_embedding_and_dense_layers(self):
    model = Sequential([
        InputLayer(input_shape=(10,), dtype='int32'),
        Embedding(1024, 32),
        Flatten(),
        Dense(8),
    ])
    dataset = [
        np.arange(40, dtype=np.int32).reshape(4, 10) % 1024,
        (np.arange(40, 80, dtype=np.int32).reshape(4, 10) % 1024),
    ]
    model(dataset[0])

    stats = collect_calibration_stats(
        model,
        dataset,
        CalibrationConfig(max_steps=8, max_samples=16),
    )

    self.assertIn('embedding', stats)
    self.assertIn('dense', stats)
    self.assertGreater(stats['embedding']['output_abs_max'], 0.0)
    self.assertGreater(stats['dense']['output_rms'], 0.0)
    self.assertEqual(stats['embedding']['sample_count'], 8)

  def test_summarize_model_can_merge_calibration_stats(self):
    model = Sequential([
        InputLayer(input_shape=(10,), dtype='int32'),
        Embedding(1024, 32),
        Flatten(),
        Dense(8),
    ])
    dataset = [
        np.arange(40, dtype=np.int32).reshape(4, 10) % 1024,
        (np.arange(40, 80, dtype=np.int32).reshape(4, 10) % 1024),
    ]
    model(dataset[0])
    stats = collect_calibration_stats(
        model,
        dataset,
        CalibrationConfig(max_steps=8, max_samples=16),
    )

    summaries = summarize_model(
        model,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=6.0),
        calibration_stats=stats,
    )

    self.assertLen(summaries, 2)
    self.assertIn('calibration', summaries[0])
    self.assertIn('normalized_mean_squared_error', summaries[0])
    self.assertGreaterEqual(summaries[0]['normalized_max_abs_error'], 0.0)


if __name__ == '__main__':
  test.main()
