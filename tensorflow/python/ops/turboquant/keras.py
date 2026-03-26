"""Keras integration for TurboQuant."""

import functools

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers import convolutional
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops.turboquant.config import TurboQuantConfig
from tensorflow.python.ops.turboquant.core import TurboQuantEncoding
from tensorflow.python.ops.turboquant.core import quantize_tensor


def _build_packed_kernel_weights(layer, output_channels, row_count):
  """Creates non-trainable variables for a packed TurboQuant kernel."""
  quantization_config = layer.quantization_config
  num_groups = (
      row_count + quantization_config.group_size - 1
  ) // quantization_config.group_size
  layer._packed_row_count = int(row_count)
  layer._packed_output_channels = int(output_channels)
  layer._packed_num_groups = int(num_groups)

  layer.codebooks = layer.add_weight(
      'codebooks',
      shape=[output_channels, quantization_config.levels],
      initializer='zeros',
      dtype=layer.dtype,
      trainable=False)
  layer.scales = layer.add_weight(
      'scales',
      shape=[output_channels, num_groups],
      initializer='ones',
      dtype=layer.dtype,
      trainable=False)
  layer.indices = layer.add_weight(
      'indices',
      shape=[output_channels, num_groups, quantization_config.group_size],
      initializer='zeros',
      dtype=dtypes.int32,
      trainable=False)
  layer.residual = layer.add_weight(
      'residual',
      shape=[output_channels, num_groups, quantization_config.group_size],
      initializer='zeros',
      dtype=layer.dtype,
      trainable=False)


def _assign_encoding_state(layer, encoding: TurboQuantEncoding):
  layer.codebooks.assign(encoding.codebooks)
  layer.scales.assign(encoding.scales)
  layer.indices.assign(encoding.indices)
  layer.residual.assign(encoding.residual)


def _dequantize_packed_kernel(layer):
  gathered = array_ops.gather(
      layer.codebooks, layer.indices, axis=1, batch_dims=1
  )
  grouped = (
      math_ops.cast(gathered, layer._compute_dtype_object)
      * array_ops.expand_dims(
          math_ops.cast(layer.scales, layer._compute_dtype_object), axis=-1
      )
      + math_ops.cast(layer.residual, layer._compute_dtype_object)
  )
  kernel = array_ops.transpose(grouped, perm=[1, 2, 0])
  kernel = array_ops.reshape(
      kernel,
      [
          layer._packed_num_groups * layer.quantization_config.group_size,
          layer._packed_output_channels,
      ],
  )
  return kernel[:layer._packed_row_count, :]


@generic_utils.register_keras_serializable(package='TurboQuant')
class TurboDense(Layer):
  """Inference-oriented Dense layer backed by TurboQuant packed weights."""

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               quantization_config=None,
               **kwargs):
    super(TurboDense, self).__init__(
        activity_regularizer=activity_regularizer, **kwargs
    )
    self.units = int(units) if not isinstance(units, int) else units
    if self.units < 0:
      raise ValueError(
          f'Received an invalid value for `units`, expected a positive '
          f'integer, got {units}.'
      )
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.quantization_config = TurboQuantConfig.from_dict(quantization_config)

    self.input_spec = InputSpec(min_ndim=2)
    self.supports_masking = True
    self._input_dim = None

  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or backend.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError(
          'Unable to build `TurboDense` layer with non-floating point '
          f'dtype {dtype}.'
      )

    input_shape = tensor_shape.TensorShape(input_shape)
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError(
          'The last dimension of the inputs to `TurboDense` should be '
          'defined. Found `None`.'
      )

    self._input_dim = int(last_dim)
    self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
    _build_packed_kernel_weights(self, self.units, self._input_dim)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=False)
    else:
      self.bias = None

    self.built = True

  def _encoding_from_dense(self, dense_layer: Layer) -> TurboQuantEncoding:
    weights = dense_layer.get_weights()
    if not weights:
      raise ValueError(
          f'Layer `{dense_layer.name}` must be built before quantization.'
      )
    kernel = weights[0]
    return quantize_tensor(kernel, self.quantization_config)

  def quantize_from_dense(self, dense_layer: Layer):
    encoding = self._encoding_from_dense(dense_layer)
    if not self.built:
      self.build([None, encoding.original_shape[0]])

    _assign_encoding_state(self, encoding)

    if self.use_bias and dense_layer.use_bias:
      self.bias.assign(dense_layer.get_weights()[1])

  def dequantized_kernel(self):
    return _dequantize_packed_kernel(self)

  def call(self, inputs):
    if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
      inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)

    kernel = self.dequantized_kernel()
    rank = inputs.shape.rank
    if rank == 2 or rank is None:
      outputs = math_ops.matmul(inputs, kernel)
    else:
      outputs = standard_ops.tensordot(inputs, kernel, [[rank - 1], [0]])
      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        outputs.set_shape(shape[:-1] + [self.units])

    if self.use_bias:
      outputs = nn_ops.bias_add(outputs, self.bias)

    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: '
          f'{input_shape}'
      )
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = super(TurboDense, self).get_config()
    config.update({
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint),
        'quantization_config': self.quantization_config.to_dict(),
    })
    return config


@generic_utils.register_keras_serializable(package='TurboQuant')
class TurboConv2D(convolutional.Conv2D):
  """Inference-oriented Conv2D layer backed by TurboQuant packed weights."""

  def __init__(self, *args, quantization_config=None, **kwargs):
    super(TurboConv2D, self).__init__(*args, **kwargs)
    self.quantization_config = TurboQuantConfig.from_dict(quantization_config)
    self._kernel_shape = None

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_channel = self._get_input_channel(input_shape)
    if input_channel % self.groups != 0:
      raise ValueError(
          'The number of input channels must be evenly divisible by the number '
          'of groups. Received groups={}, but the input has {} channels '
          '(full input shape is {}).'.format(
              self.groups, input_channel, input_shape))

    self._kernel_shape = self.kernel_size + (
        input_channel // self.groups,
        self.filters,
    )
    row_count = 1
    for dim in self._kernel_shape[:-1]:
      row_count *= int(dim)

    _build_packed_kernel_weights(self, self.filters, row_count)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=False,
          dtype=self.dtype)
    else:
      self.bias = None

    channel_axis = self._get_channel_axis()
    self.input_spec = InputSpec(
        min_ndim=self.rank + 2, axes={channel_axis: input_channel}
    )

    if self.padding == 'causal':
      tf_padding = 'VALID'
    elif isinstance(self.padding, str):
      tf_padding = self.padding.upper()
    else:
      tf_padding = self.padding
    tf_dilations = list(self.dilation_rate)
    tf_strides = list(self.strides)

    self._convolution_op = functools.partial(
        nn_ops.convolution_v2,
        strides=tf_strides,
        padding=tf_padding,
        dilations=tf_dilations,
        data_format=self._tf_data_format,
        name=self.__class__.__name__)
    self.built = True

  def _encoding_from_conv2d(self, conv_layer: Layer) -> TurboQuantEncoding:
    weights = conv_layer.get_weights()
    if not weights:
      raise ValueError(
          f'Layer `{conv_layer.name}` must be built before quantization.'
      )
    return quantize_tensor(weights[0], self.quantization_config)

  def quantize_from_conv2d(self, conv_layer: Layer):
    encoding = self._encoding_from_conv2d(conv_layer)
    if not self.built:
      self.build(conv_layer.input_shape)

    _assign_encoding_state(self, encoding)
    if self.use_bias and conv_layer.use_bias:
      self.bias.assign(conv_layer.get_weights()[1])

  def dequantized_kernel(self):
    kernel = _dequantize_packed_kernel(self)
    return array_ops.reshape(kernel, list(self._kernel_shape))

  def call(self, inputs):
    outputs = self._convolution_op(inputs, self.dequantized_kernel())

    if self.use_bias:
      output_rank = outputs.shape.rank
      if output_rank is not None and output_rank > 2 + self.rank:

        def _apply_fn(tensor):
          return nn.bias_add(tensor, self.bias, data_format=self._tf_data_format)

        outputs = conv_utils.squeeze_batch_dims(
            outputs, _apply_fn, inner_rank=self.rank + 1
        )
      else:
        outputs = nn.bias_add(
            outputs, self.bias, data_format=self._tf_data_format
        )

    if not context.executing_eagerly():
      outputs.set_shape(self.compute_output_shape(inputs.shape))

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def get_config(self):
    config = super(TurboConv2D, self).get_config()
    config['quantization_config'] = self.quantization_config.to_dict()
    return config
