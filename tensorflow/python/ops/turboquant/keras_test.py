"""Tests for TurboQuant Keras integration."""

import os

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Conv3D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import DepthwiseConv2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import SeparableConv2D
from tensorflow.python.ops.turboquant.api import export_saved_model
from tensorflow.python.ops.turboquant.api import load_saved_model
from tensorflow.python.ops.turboquant.api import quantize_model
from tensorflow.python.ops.turboquant.api import summarize_model
from tensorflow.python.ops.turboquant.config import TurboQuantConfig
from tensorflow.python.ops.turboquant.keras import TurboConv1D
from tensorflow.python.ops.turboquant.keras import TurboConv2D
from tensorflow.python.ops.turboquant.keras import TurboConv3D
from tensorflow.python.ops.turboquant.keras import TurboDense
from tensorflow.python.ops.turboquant.keras import TurboDepthwiseConv2D
from tensorflow.python.ops.turboquant.keras import TurboSeparableConv1D
from tensorflow.python.ops.turboquant.keras import TurboSeparableConv2D
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

  def test_quantize_model_replaces_conv1d_and_conv3d_layers(self):
    conv1d_model = Sequential([
        InputLayer(input_shape=(48, 16)),
        Conv1D(32, 3, padding='same', activation='relu'),
        Conv1D(16, 3, padding='same'),
    ])
    conv1d_model(np.ones((2, 48, 16), dtype=np.float32))
    quantized_conv1d = quantize_model(
        conv1d_model,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=5.0),
    )
    self.assertLen(
        [layer for layer in quantized_conv1d.layers if isinstance(layer, TurboConv1D)],
        2,
    )

    conv3d_model = Sequential([
        InputLayer(input_shape=(8, 8, 8, 2)),
        Conv3D(8, 3, padding='same', activation='relu'),
        Conv3D(4, 3, padding='same'),
    ])
    conv3d_model(np.ones((1, 8, 8, 8, 2), dtype=np.float32))
    quantized_conv3d = quantize_model(
        conv3d_model,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=5.0),
    )
    self.assertLen(
        [layer for layer in quantized_conv3d.layers if isinstance(layer, TurboConv3D)],
        2,
    )

  def test_quantize_model_replaces_depthwise_and_separable_layers(self):
    depthwise_model = Sequential([
        InputLayer(input_shape=(24, 24, 64)),
        DepthwiseConv2D(5, padding='same', depth_multiplier=1, activation='relu'),
    ])
    depthwise_model(np.ones((2, 24, 24, 64), dtype=np.float32))
    quantized_depthwise = quantize_model(
        depthwise_model,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=5.0),
    )
    self.assertLen(
        [
            layer
            for layer in quantized_depthwise.layers
            if isinstance(layer, TurboDepthwiseConv2D)
        ],
        1,
    )

    separable_1d_model = Sequential([
        InputLayer(input_shape=(96, 64)),
        SeparableConv1D(32, 5, padding='same', depth_multiplier=1, activation='relu'),
    ])
    separable_1d_model(np.ones((2, 96, 64), dtype=np.float32))
    quantized_separable_1d = quantize_model(
        separable_1d_model,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=5.0),
    )
    self.assertLen(
        [
            layer
            for layer in quantized_separable_1d.layers
            if isinstance(layer, TurboSeparableConv1D)
        ],
        1,
    )

    separable_2d_model = Sequential([
        InputLayer(input_shape=(24, 24, 64)),
        SeparableConv2D(24, 5, padding='same', depth_multiplier=1, activation='relu'),
    ])
    separable_2d_model(np.ones((2, 24, 24, 64), dtype=np.float32))
    quantized_separable_2d = quantize_model(
        separable_2d_model,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=5.0),
    )
    self.assertLen(
        [
            layer
            for layer in quantized_separable_2d.layers
            if isinstance(layer, TurboSeparableConv2D)
        ],
        1,
    )

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

  def test_quantized_depthwise_and_separable_models_stay_close_to_reference(self):
    rng = np.random.default_rng(1234)

    depthwise_inputs = rng.normal(size=(4, 24, 24, 64)).astype(np.float32)
    depthwise_model = Sequential([
        InputLayer(input_shape=(24, 24, 64)),
        DepthwiseConv2D(5, padding='same', depth_multiplier=1, activation='relu'),
        Conv2D(8, 1),
    ])
    depthwise_model(depthwise_inputs)
    quantized_depthwise = quantize_model(
        depthwise_model,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=6.0),
    )
    self.assertAllClose(
        depthwise_model(depthwise_inputs),
        quantized_depthwise(depthwise_inputs),
        atol=0.35,
        rtol=0.35,
    )

    separable_1d_inputs = rng.normal(size=(4, 96, 64)).astype(np.float32)
    separable_1d_model = Sequential([
        InputLayer(input_shape=(96, 64)),
        SeparableConv1D(32, 5, padding='same', depth_multiplier=1, activation='relu'),
    ])
    separable_1d_model(separable_1d_inputs)
    quantized_separable_1d = quantize_model(
        separable_1d_model,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=6.0),
    )
    self.assertAllClose(
        separable_1d_model(separable_1d_inputs),
        quantized_separable_1d(separable_1d_inputs),
        atol=0.35,
        rtol=0.35,
    )

    separable_2d_inputs = rng.normal(size=(2, 24, 24, 64)).astype(np.float32)
    separable_2d_model = Sequential([
        InputLayer(input_shape=(24, 24, 64)),
        SeparableConv2D(24, 5, padding='same', depth_multiplier=1, activation='relu'),
    ])
    separable_2d_model(separable_2d_inputs)
    quantized_separable_2d = quantize_model(
        separable_2d_model,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=6.0),
    )
    self.assertAllClose(
        separable_2d_model(separable_2d_inputs),
        quantized_separable_2d(separable_2d_inputs),
        atol=0.35,
        rtol=0.35,
    )

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

  def test_quantized_conv1d_and_conv3d_models_stay_close_to_reference(self):
    rng = np.random.default_rng(77)

    inputs_1d = rng.normal(size=(4, 48, 16)).astype(np.float32)
    model_1d = Sequential([
        InputLayer(input_shape=(48, 16)),
        Conv1D(24, 3, padding='same', activation='relu'),
        Conv1D(16, 3, padding='same'),
    ])
    model_1d(inputs_1d)
    quantized_1d = quantize_model(
        model_1d,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=6.0),
    )
    self.assertAllClose(
        model_1d(inputs_1d), quantized_1d(inputs_1d), atol=0.25, rtol=0.25
    )

    inputs_3d = rng.normal(size=(2, 8, 8, 8, 2)).astype(np.float32)
    model_3d = Sequential([
        InputLayer(input_shape=(8, 8, 8, 2)),
        Conv3D(8, 3, padding='same', activation='relu'),
        Conv3D(4, 3, padding='same'),
    ])
    model_3d(inputs_3d)
    quantized_3d = quantize_model(
        model_3d,
        TurboQuantConfig(num_bits=4, group_size=8, outlier_threshold=6.0),
    )
    self.assertAllClose(
        model_3d(inputs_3d), quantized_3d(inputs_3d), atol=0.3, rtol=0.3
    )

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

  def test_summarize_model_reports_depthwise_and_separable_components(self):
    model = Sequential([
        InputLayer(input_shape=(24, 24, 64)),
        DepthwiseConv2D(5, padding='same', depth_multiplier=1, activation='relu'),
        SeparableConv2D(24, 5, padding='same', depth_multiplier=1),
    ])
    model(np.ones((1, 24, 24, 64), dtype=np.float32))

    summaries = summarize_model(model, TurboQuantConfig(group_size=8))

    self.assertLen(summaries, 2)
    self.assertEqual(summaries[0]['layer_type'], 'DepthwiseConv2D')
    self.assertEqual(summaries[1]['layer_type'], 'SeparableConv2D')
    self.assertIn('kernel_components', summaries[1])
    self.assertLen(summaries[1]['kernel_components'], 2)

  def test_summarize_model_can_include_skipped_layers_with_reasons(self):
    model = Sequential([
        InputLayer(input_shape=(8,)),
        Dense(4),
    ])
    model(np.ones((1, 8), dtype=np.float32))

    summaries = summarize_model(
        model,
        TurboQuantConfig(num_bits=4, group_size=8, minimum_elements=64),
        include_skipped=True,
    )

    self.assertLen(summaries, 1)
    self.assertEqual(summaries[0]['status'], 'skipped')
    self.assertEqual(summaries[0]['reason'], 'below_minimum_elements')

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

  def test_export_saved_model_round_trip_preserves_turbo_layers(self):
    rng = np.random.default_rng(5)
    inputs = rng.normal(size=(4, 12, 12, 64)).astype(np.float32)

    model = Sequential([
        InputLayer(input_shape=(12, 12, 64)),
        DepthwiseConv2D(5, padding='same', depth_multiplier=1, activation='relu'),
        SeparableConv2D(12, 5, padding='same', depth_multiplier=1, activation='relu'),
        Flatten(),
        Dense(6),
    ])
    model(inputs)

    export_dir = os.path.join(self.get_temp_dir(), 'turboquant_saved_model')
    quantized_model = export_saved_model(
        model,
        export_dir,
        TurboQuantConfig(num_bits=4, group_size=16, outlier_threshold=6.0),
    )
    loaded_model = load_saved_model(export_dir)
    loaded_outputs = loaded_model.signatures['serving_default'](
        constant_op.constant(inputs)
    )['outputs']

    self.assertTrue(
        any(isinstance(layer, TurboDepthwiseConv2D) for layer in quantized_model.layers)
    )
    self.assertTrue(
        any(isinstance(layer, TurboSeparableConv2D) for layer in quantized_model.layers)
    )
    self.assertTrue(any(isinstance(layer, TurboDense) for layer in quantized_model.layers))
    self.assertAllClose(
        quantized_model(inputs), loaded_outputs, atol=1e-5, rtol=1e-5
    )


if __name__ == '__main__':
  test.main()
