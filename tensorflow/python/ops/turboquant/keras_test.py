"""Tests for TurboQuant Keras integration."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.ops.turboquant.api import quantize_model
from tensorflow.python.ops.turboquant.api import summarize_model
from tensorflow.python.ops.turboquant.config import TurboQuantConfig
from tensorflow.python.ops.turboquant.keras import TurboConv2D
from tensorflow.python.ops.turboquant.keras import TurboDense
from tensorflow.python.platform import test


class TurboKerasIntegrationTest(test.TestCase):

  def test_quantize_model_replaces_dense_layers(self):
    model = Sequential([
        InputLayer(input_shape=(64,)),
        Dense(64, activation='relu'),
        Dense(16),
    ])
    model(np.ones((2, 64), dtype=np.float32))

    quantized_model = quantize_model(
        model,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=5.0),
    )

    dense_like_layers = [
        layer for layer in quantized_model.layers if isinstance(layer, TurboDense)
    ]
    self.assertLen(dense_like_layers, 2)

  def test_quantize_model_replaces_conv2d_layers(self):
    model = Sequential([
        InputLayer(input_shape=(16, 16, 3)),
        Conv2D(16, 3, padding='same', activation='relu'),
        Conv2D(16, 3, padding='same'),
    ])
    model(np.ones((2, 16, 16, 3), dtype=np.float32))

    quantized_model = quantize_model(
        model,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=5.0),
    )

    conv_like_layers = [
        layer for layer in quantized_model.layers if isinstance(layer, TurboConv2D)
    ]
    self.assertLen(conv_like_layers, 2)

  def test_quantized_model_stays_close_to_reference(self):
    rng = np.random.default_rng(13)
    inputs = rng.normal(size=(8, 64)).astype(np.float32)

    model = Sequential([
        InputLayer(input_shape=(64,)),
        Dense(48, activation='relu'),
        Dense(24),
    ])
    model(inputs)

    quantized_model = quantize_model(
        model,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=6.0),
    )

    reference = model(inputs)
    quantized = quantized_model(inputs)

    self.assertAllClose(reference, quantized, atol=0.2, rtol=0.2)

  def test_quantized_conv2d_model_stays_close_to_reference(self):
    rng = np.random.default_rng(123)
    inputs = rng.normal(size=(4, 16, 16, 3)).astype(np.float32)

    model = Sequential([
        InputLayer(input_shape=(16, 16, 3)),
        Conv2D(12, 3, padding='same', activation='relu'),
        Conv2D(8, 3, padding='same', activation='relu'),
        Flatten(),
        Dense(10),
    ])
    model(inputs)

    quantized_model = quantize_model(
        model,
        TurboQuantConfig(num_bits=4, group_size=16, outlier_threshold=6.0),
    )

    reference = model(inputs)
    quantized = quantized_model(inputs)

    self.assertAllClose(reference, quantized, atol=0.35, rtol=0.35)

  def test_turbodense_runs_in_tf_function(self):
    model = Sequential([
        InputLayer(input_shape=(64,)),
        Dense(32, activation='relu'),
        Dense(16),
    ])
    model(np.ones((1, 64), dtype=np.float32))

    quantized_model = quantize_model(
        model,
        TurboQuantConfig(num_bits=4, group_size=4, outlier_threshold=6.0),
    )

    @def_function.function
    def run(inputs):
      return quantized_model(inputs)

    outputs = run(np.ones((3, 64), dtype=np.float32))
    self.assertEqual(outputs.shape.as_list(), [3, 16])

  def test_summarize_model_reports_dense_layers(self):
    model = Sequential([
        InputLayer(input_shape=(128,)),
        Dense(64, activation='relu'),
        Dense(32),
    ])
    model(np.ones((1, 128), dtype=np.float32))

    summaries = summarize_model(model, TurboQuantConfig(group_size=8))

    self.assertLen(summaries, 2)
    self.assertTrue(all(summary['compression_ratio'] > 1.0 for summary in summaries))

  def test_summarize_model_reports_conv2d_layers(self):
    model = Sequential([
        InputLayer(input_shape=(16, 16, 3)),
        Conv2D(16, 3, padding='same', activation='relu'),
        Conv2D(16, 3, padding='same'),
    ])
    model(np.ones((1, 16, 16, 3), dtype=np.float32))

    summaries = summarize_model(model, TurboQuantConfig(group_size=8))

    self.assertLen(summaries, 2)
    self.assertTrue(all(summary['layer_type'] == 'Conv2D' for summary in summaries))

  def test_small_layers_are_left_unmodified_when_not_profitable(self):
    model = Sequential([
        InputLayer(input_shape=(8,)),
        Dense(4),
    ])
    model(np.ones((1, 8), dtype=np.float32))

    quantized_model = quantize_model(
        model,
        TurboQuantConfig(num_bits=4, group_size=8, minimum_elements=1),
    )

    self.assertIsInstance(quantized_model.layers[-1], Dense)


if __name__ == '__main__':
  test.main()
