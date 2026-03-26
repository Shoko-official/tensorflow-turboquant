"""Keras integration for TurboQuant."""

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
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops.turboquant.config import TurboQuantConfig
from tensorflow.python.ops.turboquant.core import TurboQuantEncoding
from tensorflow.python.ops.turboquant.core import quantize_tensor


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
    self._num_groups = (
        self._input_dim + self.quantization_config.group_size - 1
    ) // self.quantization_config.group_size
    self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

    self.codebooks = self.add_weight(
        'codebooks',
        shape=[self.units, self.quantization_config.levels],
        initializer='zeros',
        dtype=self.dtype,
        trainable=False)
    self.scales = self.add_weight(
        'scales',
        shape=[self.units, self._num_groups],
        initializer='ones',
        dtype=self.dtype,
        trainable=False)
    self.indices = self.add_weight(
        'indices',
        shape=[self.units, self._num_groups, self.quantization_config.group_size],
        initializer='zeros',
        dtype=dtypes.int32,
        trainable=False)
    self.residual = self.add_weight(
        'residual',
        shape=[self.units, self._num_groups, self.quantization_config.group_size],
        initializer='zeros',
        dtype=self.dtype,
        trainable=False)
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

    self.codebooks.assign(encoding.codebooks)
    self.scales.assign(encoding.scales)
    self.indices.assign(encoding.indices)
    self.residual.assign(encoding.residual)

    if self.use_bias and dense_layer.use_bias:
      self.bias.assign(dense_layer.get_weights()[1])

  def dequantized_kernel(self):
    gathered = array_ops.gather(
        self.codebooks, self.indices, axis=1, batch_dims=1
    )
    grouped = (
        math_ops.cast(gathered, self._compute_dtype_object)
        * array_ops.expand_dims(
            math_ops.cast(self.scales, self._compute_dtype_object), axis=-1
        )
        + math_ops.cast(self.residual, self._compute_dtype_object)
    )
    kernel = array_ops.transpose(grouped, perm=[1, 2, 0])
    kernel = array_ops.reshape(
        kernel,
        [
            self._num_groups * self.quantization_config.group_size,
            self.units,
        ],
    )
    return kernel[:self._input_dim, :]

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
