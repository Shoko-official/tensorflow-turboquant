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
from tensorflow.python.keras.layers import embeddings as embeddings_layers
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops.turboquant.config import TurboQuantConfig
from tensorflow.python.ops.turboquant.core import TurboQuantEncoding
from tensorflow.python.ops.turboquant.core import quantize_tensor


def _packed_attr_name(prefix, name):
  return f'{prefix}_{name}' if prefix else name


def _index_dtype_for_config(quantization_config):
  if quantization_config.levels <= 256:
    return dtypes.uint8
  return dtypes.int32


def _set_packed_attr(layer, prefix, name, value):
  setattr(layer, _packed_attr_name(prefix, name), value)


def _get_packed_attr(layer, prefix, name):
  return getattr(layer, _packed_attr_name(prefix, name))


def _build_packed_kernel_weights(layer, output_channels, row_count, prefix=None):
  """Creates non-trainable variables for a packed TurboQuant kernel."""
  quantization_config = layer.quantization_config
  index_dtype = _index_dtype_for_config(quantization_config)
  num_groups = (
      row_count + quantization_config.group_size - 1
  ) // quantization_config.group_size
  _set_packed_attr(layer, prefix, '_packed_row_count', int(row_count))
  _set_packed_attr(
      layer, prefix, '_packed_output_channels', int(output_channels)
  )
  _set_packed_attr(layer, prefix, '_packed_num_groups', int(num_groups))

  _set_packed_attr(
      layer,
      prefix,
      'codebooks',
      layer.add_weight(
          _packed_attr_name(prefix, 'codebooks'),
          shape=[output_channels, quantization_config.levels],
          initializer='zeros',
          dtype=layer.dtype,
          trainable=False),
  )
  _set_packed_attr(
      layer,
      prefix,
      'scales',
      layer.add_weight(
          _packed_attr_name(prefix, 'scales'),
          shape=[output_channels, num_groups],
          initializer='ones',
          dtype=layer.dtype,
          trainable=False),
  )
  _set_packed_attr(
      layer,
      prefix,
      'indices',
      layer.add_weight(
          _packed_attr_name(prefix, 'indices'),
          shape=[output_channels, num_groups, quantization_config.group_size],
          initializer='zeros',
          dtype=index_dtype,
          trainable=False),
  )
  _set_packed_attr(
      layer,
      prefix,
      'residual',
      layer.add_weight(
          _packed_attr_name(prefix, 'residual'),
          shape=[output_channels, num_groups, quantization_config.group_size],
          initializer='zeros',
          dtype=layer.dtype,
          trainable=False),
  )


def _assign_encoding_state(layer, encoding: TurboQuantEncoding, prefix=None):
  _get_packed_attr(layer, prefix, 'codebooks').assign(encoding.codebooks)
  _get_packed_attr(layer, prefix, 'scales').assign(encoding.scales)
  indices_var = _get_packed_attr(layer, prefix, 'indices')
  _get_packed_attr(layer, prefix, 'indices').assign(
      encoding.indices.astype(indices_var.dtype.as_numpy_dtype, copy=False)
  )
  _get_packed_attr(layer, prefix, 'residual').assign(encoding.residual)


def _dequantize_packed_kernel(layer, prefix=None):
  codebooks = _get_packed_attr(layer, prefix, 'codebooks')
  indices = _get_packed_attr(layer, prefix, 'indices')
  scales = _get_packed_attr(layer, prefix, 'scales')
  residual = _get_packed_attr(layer, prefix, 'residual')
  packed_output_channels = _get_packed_attr(
      layer, prefix, '_packed_output_channels'
  )
  packed_num_groups = _get_packed_attr(layer, prefix, '_packed_num_groups')
  packed_row_count = _get_packed_attr(layer, prefix, '_packed_row_count')

  gathered = array_ops.gather(
      codebooks, math_ops.cast(indices, dtypes.int32), axis=1, batch_dims=1
  )
  grouped = (
      math_ops.cast(gathered, layer._compute_dtype_object)
      * array_ops.expand_dims(
          math_ops.cast(scales, layer._compute_dtype_object), axis=-1
      )
      + math_ops.cast(residual, layer._compute_dtype_object)
  )
  kernel = array_ops.transpose(grouped, perm=[1, 2, 0])
  kernel = array_ops.reshape(
      kernel,
      [
          packed_num_groups * layer.quantization_config.group_size,
          packed_output_channels,
      ],
  )
  return kernel[:packed_row_count, :]


def _kernel_row_count(kernel_shape):
  row_count = 1
  for dim in kernel_shape[:-1]:
    row_count *= int(dim)
  return row_count


def _depthwise_row_count(kernel_shape):
  row_count = 1
  for dim in kernel_shape[:-2]:
    row_count *= int(dim)
  return row_count


def _reshape_depthwise_kernel_for_quantization(kernel):
  row_count = _depthwise_row_count(kernel.shape)
  return kernel.reshape(row_count, kernel.shape[-2] * kernel.shape[-1])


def _dequantize_depthwise_kernel(layer, prefix=None):
  kernel = _dequantize_packed_kernel(layer, prefix=prefix)
  return array_ops.reshape(kernel, list(layer._depthwise_kernel_shape))


def _build_conv_state(layer, input_shape):
  """Initializes shared TurboQuant convolution state."""
  input_shape = tensor_shape.TensorShape(input_shape)
  input_channel = layer._get_input_channel(input_shape)
  if input_channel % layer.groups != 0:
    raise ValueError(
        'The number of input channels must be evenly divisible by the number '
        'of groups. Received groups={}, but the input has {} channels '
        '(full input shape is {}).'.format(layer.groups, input_channel, input_shape)
    )

  layer._kernel_shape = layer.kernel_size + (
      input_channel // layer.groups,
      layer.filters,
  )
  _build_packed_kernel_weights(
      layer,
      output_channels=layer.filters,
      row_count=_kernel_row_count(layer._kernel_shape),
  )
  if layer.use_bias:
    layer.bias = layer.add_weight(
        name='bias',
        shape=(layer.filters,),
        initializer=layer.bias_initializer,
        regularizer=layer.bias_regularizer,
        constraint=layer.bias_constraint,
        trainable=False,
        dtype=layer.dtype)
  else:
    layer.bias = None

  channel_axis = layer._get_channel_axis()
  layer.input_spec = InputSpec(
      min_ndim=layer.rank + 2, axes={channel_axis: input_channel}
  )

  if layer.padding == 'causal':
    tf_padding = 'VALID'
  elif isinstance(layer.padding, str):
    tf_padding = layer.padding.upper()
  else:
    tf_padding = layer.padding
  tf_dilations = list(layer.dilation_rate)
  tf_strides = list(layer.strides)

  layer._convolution_op = functools.partial(
      nn_ops.convolution_v2,
      strides=tf_strides,
      padding=tf_padding,
      dilations=tf_dilations,
      data_format=layer._tf_data_format,
      name=layer.__class__.__name__)
  layer.built = True


def _build_conv_transpose_state(layer, input_shape):
  """Initializes shared TurboQuant transposed convolution state."""
  input_shape = tensor_shape.TensorShape(input_shape)
  expected_rank = layer.rank + 2
  if len(input_shape) != expected_rank:
    raise ValueError(
        f'Inputs should have rank {expected_rank}. Received input shape: '
        f'{input_shape}.'
    )
  channel_axis = layer._get_channel_axis()
  input_dim = tensor_shape.dimension_value(input_shape[channel_axis])
  if input_dim is None:
    raise ValueError(
        'The channel dimension of the inputs should be defined. '
        'Found `None`.'
    )
  input_dim = int(input_dim)
  layer._kernel_shape = layer.kernel_size + (layer.filters, input_dim)
  _build_packed_kernel_weights(
      layer,
      output_channels=input_dim,
      row_count=_kernel_row_count(layer._kernel_shape),
  )
  if layer.use_bias:
    layer.bias = layer.add_weight(
        name='bias',
        shape=(layer.filters,),
        initializer=layer.bias_initializer,
        regularizer=layer.bias_regularizer,
        constraint=layer.bias_constraint,
        trainable=False,
        dtype=layer.dtype)
  else:
    layer.bias = None
  layer.input_spec = InputSpec(ndim=expected_rank, axes={channel_axis: input_dim})
  layer.built = True


def _build_embedding_state(layer):
  """Initializes shared TurboQuant embedding state."""
  _build_packed_kernel_weights(
      layer,
      output_channels=layer.output_dim,
      row_count=layer.input_dim,
  )
  layer.built = True


def _build_depthwise_conv_state(layer, input_shape):
  """Initializes shared TurboQuant depthwise convolution state."""
  input_shape = tensor_shape.TensorShape(input_shape)
  if len(input_shape) < 4:
    raise ValueError(
        'Inputs to `DepthwiseConv2D` should have rank 4. '
        f'Received input shape: {input_shape}.'
    )
  channel_axis = layer._get_channel_axis()
  input_dim = tensor_shape.dimension_value(input_shape[channel_axis])
  if input_dim is None:
    raise ValueError(
        'The channel dimension of the inputs to `DepthwiseConv2D` '
        'should be defined. Found `None`.'
    )

  input_dim = int(input_dim)
  layer._depthwise_kernel_shape = (
      layer.kernel_size[0],
      layer.kernel_size[1],
      input_dim,
      layer.depth_multiplier,
  )
  _build_packed_kernel_weights(
      layer,
      output_channels=input_dim * layer.depth_multiplier,
      row_count=_depthwise_row_count(layer._depthwise_kernel_shape),
  )
  if layer.use_bias:
    layer.bias = layer.add_weight(
        name='bias',
        shape=(input_dim * layer.depth_multiplier,),
        initializer=layer.bias_initializer,
        regularizer=layer.bias_regularizer,
        constraint=layer.bias_constraint,
        trainable=False,
        dtype=layer.dtype)
  else:
    layer.bias = None

  layer.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
  layer.built = True


def _build_separable_conv_state(layer, input_shape):
  """Initializes shared TurboQuant separable convolution state."""
  input_shape = tensor_shape.TensorShape(input_shape)
  channel_axis = layer._get_channel_axis()
  input_dim = tensor_shape.dimension_value(input_shape[channel_axis])
  if input_dim is None:
    raise ValueError(
        'The channel dimension of the inputs should be defined. '
        'Found `None`.'
    )

  input_dim = int(input_dim)
  layer.input_spec = InputSpec(
      ndim=layer.rank + 2, axes={channel_axis: input_dim}
  )
  layer._depthwise_kernel_shape = layer.kernel_size + (
      input_dim,
      layer.depth_multiplier,
  )
  layer._pointwise_kernel_shape = (1,) * layer.rank + (
      input_dim * layer.depth_multiplier,
      layer.filters,
  )
  _build_packed_kernel_weights(
      layer,
      output_channels=input_dim * layer.depth_multiplier,
      row_count=_depthwise_row_count(layer._depthwise_kernel_shape),
      prefix='depthwise',
  )
  _build_packed_kernel_weights(
      layer,
      output_channels=layer.filters,
      row_count=_kernel_row_count(layer._pointwise_kernel_shape),
      prefix='pointwise',
  )
  if layer.use_bias:
    layer.bias = layer.add_weight(
        name='bias',
        shape=(layer.filters,),
        initializer=layer.bias_initializer,
        regularizer=layer.bias_regularizer,
        constraint=layer.bias_constraint,
        trainable=False,
        dtype=layer.dtype)
  else:
    layer.bias = None
  layer.built = True


def _quantize_from_conv(layer, conv_layer: Layer):
  weights = conv_layer.get_weights()
  if not weights:
    raise ValueError(
        f'Layer `{conv_layer.name}` must be built before quantization.'
    )
  encoding = quantize_tensor(weights[0], layer.quantization_config)
  if not layer.built:
    layer.build(conv_layer.input_shape)
  _assign_encoding_state(layer, encoding)
  if layer.use_bias and conv_layer.use_bias:
    layer.bias.assign(weights[1])


def _quantize_from_embedding(layer, embedding_layer: Layer):
  weights = embedding_layer.get_weights()
  if not weights:
    raise ValueError(
        f'Layer `{embedding_layer.name}` must be built before quantization.'
    )
  encoding = quantize_tensor(weights[0], layer.quantization_config)
  if not layer.built:
    layer.build()
  _assign_encoding_state(layer, encoding)


def _quantize_from_depthwise(layer, conv_layer: Layer):
  weights = conv_layer.get_weights()
  if not weights:
    raise ValueError(
        f'Layer `{conv_layer.name}` must be built before quantization.'
    )
  depthwise_kernel = weights[0]
  encoding = quantize_tensor(
      _reshape_depthwise_kernel_for_quantization(depthwise_kernel),
      layer.quantization_config,
  )
  if not layer.built:
    layer.build(conv_layer.input_shape)
  _assign_encoding_state(layer, encoding)
  if layer.use_bias and conv_layer.use_bias:
    layer.bias.assign(weights[1])


def _quantize_from_separable(layer, conv_layer: Layer):
  weights = conv_layer.get_weights()
  if len(weights) < 2:
    raise ValueError(
        f'Layer `{conv_layer.name}` must be built before quantization.'
    )
  if not layer.built:
    layer.build(conv_layer.input_shape)
  depthwise_encoding = quantize_tensor(
      _reshape_depthwise_kernel_for_quantization(weights[0]),
      layer.quantization_config,
  )
  pointwise_encoding = quantize_tensor(weights[1], layer.quantization_config)
  _assign_encoding_state(layer, depthwise_encoding, prefix='depthwise')
  _assign_encoding_state(layer, pointwise_encoding, prefix='pointwise')
  if layer.use_bias and conv_layer.use_bias:
    layer.bias.assign(weights[2])


def _call_quantized_conv(layer, inputs):
  if layer._is_causal:
    inputs = array_ops.pad(inputs, layer._compute_causal_padding(inputs))

  outputs = layer._convolution_op(inputs, layer.dequantized_kernel())
  if layer.use_bias:
    output_rank = outputs.shape.rank
    if layer.rank == 1 and layer._channels_first:
      bias = array_ops.reshape(layer.bias, (1, layer.filters, 1))
      outputs += bias
    elif output_rank is not None and output_rank > 2 + layer.rank:

      def _apply_fn(tensor):
        return nn.bias_add(tensor, layer.bias, data_format=layer._tf_data_format)

      outputs = conv_utils.squeeze_batch_dims(
          outputs, _apply_fn, inner_rank=layer.rank + 1
      )
    else:
      outputs = nn.bias_add(
          outputs, layer.bias, data_format=layer._tf_data_format
      )

  if not context.executing_eagerly():
    outputs.set_shape(layer.compute_output_shape(inputs.shape))

  if layer.activation is not None:
    return layer.activation(outputs)
  return outputs


def _call_quantized_conv1d_transpose(layer, inputs):
  inputs_shape = array_ops.shape(inputs)
  batch_size = inputs_shape[0]
  if layer.data_format == 'channels_first':
    t_axis = 2
  else:
    t_axis = 1

  length = inputs_shape[t_axis]
  if layer.output_padding is None:
    output_padding = None
  else:
    output_padding = layer.output_padding[0]

  out_length = conv_utils.deconv_output_length(
      length,
      layer.kernel_size[0],
      padding=layer.padding,
      output_padding=output_padding,
      stride=layer.strides[0],
      dilation=layer.dilation_rate[0],
  )
  if layer.data_format == 'channels_first':
    output_shape = (batch_size, layer.filters, out_length)
  else:
    output_shape = (batch_size, out_length, layer.filters)
  data_format = conv_utils.convert_data_format(layer.data_format, ndim=3)

  output_shape_tensor = array_ops_stack.stack(output_shape)
  outputs = nn_ops.conv1d_transpose(
      inputs,
      layer.dequantized_kernel(),
      output_shape_tensor,
      strides=layer.strides,
      padding=layer.padding.upper(),
      data_format=data_format,
      dilations=layer.dilation_rate,
  )

  if not context.executing_eagerly():
    outputs.set_shape(layer.compute_output_shape(inputs.shape))

  if layer.use_bias:
    outputs = nn.bias_add(outputs, layer.bias, data_format=data_format)

  if layer.activation is not None:
    return layer.activation(outputs)
  return outputs


def _call_quantized_conv2d_transpose(layer, inputs):
  inputs_shape = array_ops.shape(inputs)
  batch_size = inputs_shape[0]
  if layer.data_format == 'channels_first':
    h_axis, w_axis = 2, 3
  else:
    h_axis, w_axis = 1, 2

  height, width = None, None
  if inputs.shape.rank is not None:
    dims = inputs.shape.as_list()
    height = dims[h_axis]
    width = dims[w_axis]
  height = height if height is not None else inputs_shape[h_axis]
  width = width if width is not None else inputs_shape[w_axis]

  kernel_h, kernel_w = layer.kernel_size
  stride_h, stride_w = layer.strides
  if layer.output_padding is None:
    out_pad_h = out_pad_w = None
  else:
    out_pad_h, out_pad_w = layer.output_padding

  out_height = conv_utils.deconv_output_length(
      height,
      kernel_h,
      padding=layer.padding,
      output_padding=out_pad_h,
      stride=stride_h,
      dilation=layer.dilation_rate[0],
  )
  out_width = conv_utils.deconv_output_length(
      width,
      kernel_w,
      padding=layer.padding,
      output_padding=out_pad_w,
      stride=stride_w,
      dilation=layer.dilation_rate[1],
  )
  if layer.data_format == 'channels_first':
    output_shape = (batch_size, layer.filters, out_height, out_width)
  else:
    output_shape = (batch_size, out_height, out_width, layer.filters)

  output_shape_tensor = array_ops_stack.stack(output_shape)
  outputs = backend.conv2d_transpose(
      inputs,
      layer.dequantized_kernel(),
      output_shape_tensor,
      strides=layer.strides,
      padding=layer.padding,
      data_format=layer.data_format,
      dilation_rate=layer.dilation_rate,
  )

  if not context.executing_eagerly():
    outputs.set_shape(layer.compute_output_shape(inputs.shape))

  if layer.use_bias:
    outputs = nn.bias_add(
        outputs,
        layer.bias,
        data_format=conv_utils.convert_data_format(layer.data_format, ndim=4),
    )
  if layer.activation is not None:
    return layer.activation(outputs)
  return outputs


def _call_quantized_conv3d_transpose(layer, inputs):
  inputs_shape = array_ops.shape(inputs)
  batch_size = inputs_shape[0]
  if layer.data_format == 'channels_first':
    d_axis, h_axis, w_axis = 2, 3, 4
  else:
    d_axis, h_axis, w_axis = 1, 2, 3

  depth = inputs_shape[d_axis]
  height = inputs_shape[h_axis]
  width = inputs_shape[w_axis]

  kernel_d, kernel_h, kernel_w = layer.kernel_size
  stride_d, stride_h, stride_w = layer.strides
  if layer.output_padding is None:
    out_pad_d = out_pad_h = out_pad_w = None
  else:
    out_pad_d, out_pad_h, out_pad_w = layer.output_padding

  out_depth = conv_utils.deconv_output_length(
      depth,
      kernel_d,
      padding=layer.padding,
      output_padding=out_pad_d,
      stride=stride_d,
  )
  out_height = conv_utils.deconv_output_length(
      height,
      kernel_h,
      padding=layer.padding,
      output_padding=out_pad_h,
      stride=stride_h,
  )
  out_width = conv_utils.deconv_output_length(
      width,
      kernel_w,
      padding=layer.padding,
      output_padding=out_pad_w,
      stride=stride_w,
  )
  if layer.data_format == 'channels_first':
    output_shape = (batch_size, layer.filters, out_depth, out_height, out_width)
    strides = (1, 1, stride_d, stride_h, stride_w)
  else:
    output_shape = (batch_size, out_depth, out_height, out_width, layer.filters)
    strides = (1, stride_d, stride_h, stride_w, 1)

  output_shape_tensor = array_ops_stack.stack(output_shape)
  outputs = nn.conv3d_transpose(
      inputs,
      layer.dequantized_kernel(),
      output_shape_tensor,
      strides,
      data_format=conv_utils.convert_data_format(layer.data_format, ndim=5),
      padding=layer.padding.upper(),
  )

  if not context.executing_eagerly():
    outputs.set_shape(layer.compute_output_shape(inputs.shape))

  if layer.use_bias:
    outputs = nn.bias_add(
        outputs,
        layer.bias,
        data_format=conv_utils.convert_data_format(layer.data_format, ndim=4),
    )

  if layer.activation is not None:
    return layer.activation(outputs)
  return outputs


def _call_quantized_depthwise_conv(layer, inputs):
  outputs = backend.depthwise_conv2d(
      inputs,
      layer.dequantized_kernel(),
      strides=layer.strides,
      padding=layer.padding,
      dilation_rate=layer.dilation_rate,
      data_format=layer.data_format)
  if layer.use_bias:
    outputs = nn.bias_add(
        outputs,
        layer.bias,
        data_format=conv_utils.convert_data_format(layer.data_format, ndim=4))
  if layer.activation is not None:
    return layer.activation(outputs)
  return outputs


def _call_quantized_separable_conv1d(layer, inputs):
  if layer.padding == 'causal':
    inputs = array_ops.pad(inputs, layer._compute_causal_padding(inputs))
  if layer.data_format == 'channels_last':
    strides = (1,) + layer.strides * 2 + (1,)
    spatial_start_dim = 1
  else:
    strides = (1, 1) + layer.strides * 2
    spatial_start_dim = 2

  inputs = array_ops.expand_dims(inputs, spatial_start_dim)
  depthwise_kernel = array_ops.expand_dims(
      layer.dequantized_depthwise_kernel(), 0
  )
  pointwise_kernel = array_ops.expand_dims(
      layer.dequantized_pointwise_kernel(), 0
  )
  dilation_rate = (1,) + layer.dilation_rate
  if layer.padding == 'causal':
    op_padding = 'valid'
  else:
    op_padding = layer.padding
  outputs = nn.separable_conv2d(
      inputs,
      depthwise_kernel,
      pointwise_kernel,
      strides=strides,
      padding=op_padding.upper(),
      rate=dilation_rate,
      data_format=conv_utils.convert_data_format(layer.data_format, ndim=4))

  if layer.use_bias:
    outputs = nn.bias_add(
        outputs,
        layer.bias,
        data_format=conv_utils.convert_data_format(layer.data_format, ndim=4))

  outputs = array_ops.squeeze(outputs, [spatial_start_dim])
  if layer.activation is not None:
    return layer.activation(outputs)
  return outputs


def _call_quantized_separable_conv2d(layer, inputs):
  if layer.data_format == 'channels_last':
    strides = (1,) + layer.strides + (1,)
  else:
    strides = (1, 1) + layer.strides
  outputs = nn.separable_conv2d(
      inputs,
      layer.dequantized_depthwise_kernel(),
      layer.dequantized_pointwise_kernel(),
      strides=strides,
      padding=layer.padding.upper(),
      rate=layer.dilation_rate,
      data_format=conv_utils.convert_data_format(layer.data_format, ndim=4))

  if layer.use_bias:
    outputs = nn.bias_add(
        outputs,
        layer.bias,
        data_format=conv_utils.convert_data_format(layer.data_format, ndim=4))
  if layer.activation is not None:
    return layer.activation(outputs)
  return outputs


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
class TurboEmbedding(embeddings_layers.Embedding):
  """Inference-oriented Embedding layer backed by TurboQuant packed weights."""

  def __init__(self, *args, quantization_config=None, **kwargs):
    super(TurboEmbedding, self).__init__(*args, **kwargs)
    self.quantization_config = TurboQuantConfig.from_dict(quantization_config)

  def build(self, input_shape=None):
    _build_embedding_state(self)

  def quantize_from_embedding(self, embedding_layer: Layer):
    _quantize_from_embedding(self, embedding_layer)

  def dequantized_embeddings(self):
    return _dequantize_packed_kernel(self)

  def call(self, inputs):
    dtype = backend.dtype(inputs)
    if dtype != 'int32' and dtype != 'int64':
      inputs = math_ops.cast(inputs, 'int32')
    outputs = embedding_ops.embedding_lookup_v2(
        self.dequantized_embeddings(), inputs
    )
    if self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype:
      outputs = math_ops.cast(outputs, self._dtype_policy.compute_dtype)
    return outputs

  def get_config(self):
    config = super(TurboEmbedding, self).get_config()
    config['quantization_config'] = self.quantization_config.to_dict()
    return config


@generic_utils.register_keras_serializable(package='TurboQuant')
class TurboConv1D(convolutional.Conv1D):
  """Inference-oriented Conv1D layer backed by TurboQuant packed weights."""

  def __init__(self, *args, quantization_config=None, **kwargs):
    super(TurboConv1D, self).__init__(*args, **kwargs)
    self.quantization_config = TurboQuantConfig.from_dict(quantization_config)
    self._kernel_shape = None

  def build(self, input_shape):
    _build_conv_state(self, input_shape)

  def quantize_from_conv(self, conv_layer: Layer):
    _quantize_from_conv(self, conv_layer)

  def dequantized_kernel(self):
    kernel = _dequantize_packed_kernel(self)
    return array_ops.reshape(kernel, list(self._kernel_shape))

  def call(self, inputs):
    return _call_quantized_conv(self, inputs)

  def get_config(self):
    config = super(TurboConv1D, self).get_config()
    config['quantization_config'] = self.quantization_config.to_dict()
    return config


@generic_utils.register_keras_serializable(package='TurboQuant')
class TurboConv2D(convolutional.Conv2D):
  """Inference-oriented Conv2D layer backed by TurboQuant packed weights."""

  def __init__(self, *args, quantization_config=None, **kwargs):
    super(TurboConv2D, self).__init__(*args, **kwargs)
    self.quantization_config = TurboQuantConfig.from_dict(quantization_config)
    self._kernel_shape = None

  def build(self, input_shape):
    _build_conv_state(self, input_shape)

  def quantize_from_conv(self, conv_layer: Layer):
    _quantize_from_conv(self, conv_layer)

  def dequantized_kernel(self):
    kernel = _dequantize_packed_kernel(self)
    return array_ops.reshape(kernel, list(self._kernel_shape))

  def call(self, inputs):
    return _call_quantized_conv(self, inputs)

  def get_config(self):
    config = super(TurboConv2D, self).get_config()
    config['quantization_config'] = self.quantization_config.to_dict()
    return config


@generic_utils.register_keras_serializable(package='TurboQuant')
class TurboConv3D(convolutional.Conv3D):
  """Inference-oriented Conv3D layer backed by TurboQuant packed weights."""

  def __init__(self, *args, quantization_config=None, **kwargs):
    super(TurboConv3D, self).__init__(*args, **kwargs)
    self.quantization_config = TurboQuantConfig.from_dict(quantization_config)
    self._kernel_shape = None

  def build(self, input_shape):
    _build_conv_state(self, input_shape)

  def quantize_from_conv(self, conv_layer: Layer):
    _quantize_from_conv(self, conv_layer)

  def dequantized_kernel(self):
    kernel = _dequantize_packed_kernel(self)
    return array_ops.reshape(kernel, list(self._kernel_shape))

  def call(self, inputs):
    return _call_quantized_conv(self, inputs)

  def get_config(self):
    config = super(TurboConv3D, self).get_config()
    config['quantization_config'] = self.quantization_config.to_dict()
    return config


@generic_utils.register_keras_serializable(package='TurboQuant')
class TurboConv1DTranspose(convolutional.Conv1DTranspose):
  """Inference-oriented Conv1DTranspose backed by TurboQuant weights."""

  def __init__(self, *args, quantization_config=None, **kwargs):
    super(TurboConv1DTranspose, self).__init__(*args, **kwargs)
    self.quantization_config = TurboQuantConfig.from_dict(quantization_config)
    self._kernel_shape = None

  def build(self, input_shape):
    _build_conv_transpose_state(self, input_shape)

  def quantize_from_conv(self, conv_layer: Layer):
    _quantize_from_conv(self, conv_layer)

  def dequantized_kernel(self):
    kernel = _dequantize_packed_kernel(self)
    return array_ops.reshape(kernel, list(self._kernel_shape))

  def call(self, inputs):
    return _call_quantized_conv1d_transpose(self, inputs)

  def get_config(self):
    config = super(TurboConv1DTranspose, self).get_config()
    config['quantization_config'] = self.quantization_config.to_dict()
    return config


@generic_utils.register_keras_serializable(package='TurboQuant')
class TurboConv2DTranspose(convolutional.Conv2DTranspose):
  """Inference-oriented Conv2DTranspose backed by TurboQuant weights."""

  def __init__(self, *args, quantization_config=None, **kwargs):
    super(TurboConv2DTranspose, self).__init__(*args, **kwargs)
    self.quantization_config = TurboQuantConfig.from_dict(quantization_config)
    self._kernel_shape = None

  def build(self, input_shape):
    _build_conv_transpose_state(self, input_shape)

  def quantize_from_conv(self, conv_layer: Layer):
    _quantize_from_conv(self, conv_layer)

  def dequantized_kernel(self):
    kernel = _dequantize_packed_kernel(self)
    return array_ops.reshape(kernel, list(self._kernel_shape))

  def call(self, inputs):
    return _call_quantized_conv2d_transpose(self, inputs)

  def get_config(self):
    config = super(TurboConv2DTranspose, self).get_config()
    config['quantization_config'] = self.quantization_config.to_dict()
    return config


@generic_utils.register_keras_serializable(package='TurboQuant')
class TurboConv3DTranspose(convolutional.Conv3DTranspose):
  """Inference-oriented Conv3DTranspose backed by TurboQuant weights."""

  def __init__(self, *args, quantization_config=None, **kwargs):
    super(TurboConv3DTranspose, self).__init__(*args, **kwargs)
    self.quantization_config = TurboQuantConfig.from_dict(quantization_config)
    self._kernel_shape = None

  def build(self, input_shape):
    _build_conv_transpose_state(self, input_shape)

  def quantize_from_conv(self, conv_layer: Layer):
    _quantize_from_conv(self, conv_layer)

  def dequantized_kernel(self):
    kernel = _dequantize_packed_kernel(self)
    return array_ops.reshape(kernel, list(self._kernel_shape))

  def call(self, inputs):
    return _call_quantized_conv3d_transpose(self, inputs)

  def get_config(self):
    config = super(TurboConv3DTranspose, self).get_config()
    config['quantization_config'] = self.quantization_config.to_dict()
    return config


@generic_utils.register_keras_serializable(package='TurboQuant')
class TurboDepthwiseConv2D(convolutional.DepthwiseConv2D):
  """Inference-oriented DepthwiseConv2D layer backed by TurboQuant."""

  def __init__(self, *args, quantization_config=None, **kwargs):
    super(TurboDepthwiseConv2D, self).__init__(*args, **kwargs)
    self.quantization_config = TurboQuantConfig.from_dict(quantization_config)
    self._depthwise_kernel_shape = None

  def build(self, input_shape):
    _build_depthwise_conv_state(self, input_shape)

  def quantize_from_conv(self, conv_layer: Layer):
    _quantize_from_depthwise(self, conv_layer)

  def dequantized_kernel(self):
    return _dequantize_depthwise_kernel(self)

  def call(self, inputs):
    return _call_quantized_depthwise_conv(self, inputs)

  def get_config(self):
    config = super(TurboDepthwiseConv2D, self).get_config()
    config['quantization_config'] = self.quantization_config.to_dict()
    return config


@generic_utils.register_keras_serializable(package='TurboQuant')
class TurboSeparableConv1D(convolutional.SeparableConv1D):
  """Inference-oriented SeparableConv1D layer backed by TurboQuant."""

  def __init__(self, *args, quantization_config=None, **kwargs):
    super(TurboSeparableConv1D, self).__init__(*args, **kwargs)
    self.quantization_config = TurboQuantConfig.from_dict(quantization_config)
    self._depthwise_kernel_shape = None
    self._pointwise_kernel_shape = None

  def build(self, input_shape):
    _build_separable_conv_state(self, input_shape)

  def quantize_from_conv(self, conv_layer: Layer):
    _quantize_from_separable(self, conv_layer)

  def dequantized_depthwise_kernel(self):
    return _dequantize_depthwise_kernel(self, prefix='depthwise')

  def dequantized_pointwise_kernel(self):
    kernel = _dequantize_packed_kernel(self, prefix='pointwise')
    return array_ops.reshape(kernel, list(self._pointwise_kernel_shape))

  def call(self, inputs):
    return _call_quantized_separable_conv1d(self, inputs)

  def get_config(self):
    config = super(TurboSeparableConv1D, self).get_config()
    config['quantization_config'] = self.quantization_config.to_dict()
    return config


@generic_utils.register_keras_serializable(package='TurboQuant')
class TurboSeparableConv2D(convolutional.SeparableConv2D):
  """Inference-oriented SeparableConv2D layer backed by TurboQuant."""

  def __init__(self, *args, quantization_config=None, **kwargs):
    super(TurboSeparableConv2D, self).__init__(*args, **kwargs)
    self.quantization_config = TurboQuantConfig.from_dict(quantization_config)
    self._depthwise_kernel_shape = None
    self._pointwise_kernel_shape = None

  def build(self, input_shape):
    _build_separable_conv_state(self, input_shape)

  def quantize_from_conv(self, conv_layer: Layer):
    _quantize_from_separable(self, conv_layer)

  def dequantized_depthwise_kernel(self):
    return _dequantize_depthwise_kernel(self, prefix='depthwise')

  def dequantized_pointwise_kernel(self):
    kernel = _dequantize_packed_kernel(self, prefix='pointwise')
    return array_ops.reshape(kernel, list(self._pointwise_kernel_shape))

  def call(self, inputs):
    return _call_quantized_separable_conv2d(self, inputs)

  def get_config(self):
    config = super(TurboSeparableConv2D, self).get_config()
    config['quantization_config'] = self.quantization_config.to_dict()
    return config
