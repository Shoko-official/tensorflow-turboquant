"""Tests for TurboQuant calibration utilities."""

import numpy as np

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.ops.turboquant.api import quantize_model
from tensorflow.python.ops.turboquant.api import recommend_layer_configs
from tensorflow.python.ops.turboquant.api import summarize_model
from tensorflow.python.ops.turboquant.calibration import collect_calibration_stats
from tensorflow.python.ops.turboquant.config import CalibrationConfig
from tensorflow.python.ops.turboquant.config import TurboQuantConfig
from tensorflow.python.ops.turboquant.keras import TurboDense
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

  def test_summarize_model_can_collect_calibration_stats_from_dataset(self):
    model = Sequential([
        InputLayer(input_shape=(10,), dtype='int32'),
        Embedding(1024, 32),
        Flatten(),
        Dense(8),
    ])
    dataset = [
        np.arange(40, dtype=np.int32).reshape(4, 10) % 1024,
        np.arange(40, 80, dtype=np.int32).reshape(4, 10) % 1024,
    ]
    model(dataset[0])

    summaries = summarize_model(
        model,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=6.0),
        representative_dataset=dataset,
        calibration_config=CalibrationConfig(max_steps=8, max_samples=16),
    )

    self.assertLen(summaries, 2)
    self.assertIn('calibration', summaries[0])
    self.assertGreater(summaries[0]['calibration']['output_rms'], 0.0)

  def test_quantize_model_can_skip_layers_with_activation_guidance(self):
    rng = np.random.default_rng(7)
    model = Sequential([
        InputLayer(input_shape=(128,)),
        Dense(64, use_bias=False),
    ])
    reference_inputs = rng.normal(size=(4, 128)).astype(np.float32)
    model(reference_inputs)

    weights = model.layers[-1].get_weights()
    weights[0] = rng.normal(scale=1e-4, size=weights[0].shape).astype(np.float32)
    model.layers[-1].set_weights(weights)

    representative_dataset = [
        rng.normal(size=(4, 128)).astype(np.float32),
        rng.normal(size=(4, 128)).astype(np.float32),
    ]
    quantization_config = TurboQuantConfig(
        num_bits=4,
        group_size=8,
        outlier_threshold=6.0,
        max_normalized_mean_squared_error=1e-6,
    )

    summaries = summarize_model(
        model,
        quantization_config,
        include_skipped=True,
        representative_dataset=representative_dataset,
        calibration_config=CalibrationConfig(max_steps=8, max_samples=16),
    )
    quantized_model = quantize_model(
        model,
        quantization_config,
        representative_dataset=representative_dataset,
        calibration_config=CalibrationConfig(max_steps=8, max_samples=16),
    )

    self.assertEqual(summaries[0]['status'], 'skipped')
    self.assertEqual(
        summaries[0]['reason'], 'normalized_mean_squared_error_too_high'
    )
    self.assertIsInstance(quantized_model.layers[-1], Dense)

  def test_summarize_model_return_report_contains_aggregate(self):
    model = Sequential([
        InputLayer(input_shape=(10,), dtype='int32'),
        Embedding(1024, 32),
        Flatten(),
        Dense(8),
    ])
    dataset = [
        np.arange(40, dtype=np.int32).reshape(4, 10) % 1024,
        np.arange(40, 80, dtype=np.int32).reshape(4, 10) % 1024,
    ]
    model(dataset[0])

    report = summarize_model(
        model,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=6.0),
        include_skipped=True,
        representative_dataset=dataset,
        calibration_config=CalibrationConfig(max_steps=8, max_samples=16),
        return_report=True,
    )

    self.assertIn('summaries', report)
    self.assertIn('aggregate', report)
    self.assertLen(report['summaries'], 2)
    self.assertGreaterEqual(report['aggregate']['supported_layers'], 2)
    self.assertGreaterEqual(
        report['aggregate']['effective_compression_ratio'], 1.0
    )

  def test_recommend_and_quantize_respect_layer_name_filters(self):
    model = Sequential([
        InputLayer(input_shape=(16,)),
        Dense(12, activation='relu', name='dense_a'),
        Dense(6, name='dense_b'),
    ])
    dataset = [
        np.ones((4, 16), dtype=np.float32),
        np.full((4, 16), 0.5, dtype=np.float32),
    ]
    model(dataset[0])

    recommendations = recommend_layer_configs(
        model,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=6.0),
        representative_dataset=dataset,
        calibration_config=CalibrationConfig(max_steps=8, max_samples=16),
        target_layer_names=['dense_b'],
        include_skipped=True,
    )
    by_name = {item['layer_name']: item for item in recommendations}
    self.assertEqual(by_name['dense_a']['status'], 'skipped')
    self.assertEqual(by_name['dense_a']['reason'], 'not_selected_by_filter')

    quantized_model = quantize_model(
        model,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=6.0),
        representative_dataset=dataset,
        calibration_config=CalibrationConfig(max_steps=8, max_samples=16),
        target_layer_names=['dense_b'],
    )
    self.assertIsInstance(quantized_model.get_layer('dense_a'), Dense)
    self.assertIsInstance(quantized_model.get_layer('dense_b'), TurboDense)


if __name__ == '__main__':
  test.main()
