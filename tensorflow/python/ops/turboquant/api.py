"""High-level TurboQuant APIs."""

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import convolutional as convolutional_layers
from tensorflow.python.keras.layers import core as core_layers
from tensorflow.python.ops.turboquant.config import TurboQuantConfig
from tensorflow.python.ops.turboquant.core import quantize_tensor
from tensorflow.python.ops.turboquant.core import summarize_encoding
from tensorflow.python.ops.turboquant.keras import TurboConv2D
from tensorflow.python.ops.turboquant.keras import TurboDense


def _quantizable_kernel(layer):
  weights = layer.get_weights()
  if (
      not isinstance(layer, (core_layers.Dense, convolutional_layers.Conv2D))
      or not weights
  ):
    return None
  return weights[0]


def _should_quantize_layer(layer, quantization_config: TurboQuantConfig) -> bool:
  kernel = _quantizable_kernel(layer)
  if (
      kernel is None
      or not quantization_config.should_quantize(kernel.size)
  ):
    return False

  encoding = quantize_tensor(kernel, quantization_config)
  summary = summarize_encoding(kernel, encoding)
  return summary['packed_bytes'] < summary['original_bytes']


def _clone_layer(layer, quantization_config: TurboQuantConfig):
  if not _should_quantize_layer(layer, quantization_config):
    return layer.__class__.from_config(layer.get_config())

  if isinstance(layer, core_layers.Dense):
    layer_config = layer.get_config()
    layer_config['quantization_config'] = quantization_config.to_dict()
    layer_config['trainable'] = False
    return TurboDense(**layer_config)
  if isinstance(layer, convolutional_layers.Conv2D):
    layer_config = layer.get_config()
    layer_config['quantization_config'] = quantization_config.to_dict()
    layer_config['trainable'] = False
    return TurboConv2D(**layer_config)
  return layer.__class__.from_config(layer.get_config())


def quantize_model(model, quantization_config=None):
  """Clones a Functional or Sequential model with TurboQuant wrappers."""
  quantization_config = TurboQuantConfig.from_dict(quantization_config)
  if not hasattr(model, 'layers') or not hasattr(model, 'get_layer'):
    raise ValueError(
        '`quantize_model` only supports Keras models with an explicit layer '
        'graph.'
    )
  if not model.built:
    raise ValueError('`quantize_model` expects a built model.')

  quantized_model = models.clone_model(
      model,
      clone_function=lambda layer: _clone_layer(layer, quantization_config))
  if not quantized_model.built:
    quantized_model.build(model.input_shape)

  for layer in model.layers:
    cloned_layer = quantized_model.get_layer(layer.name)
    if isinstance(cloned_layer, TurboDense):
      cloned_layer.quantize_from_dense(layer)
    elif isinstance(cloned_layer, TurboConv2D):
      cloned_layer.quantize_from_conv2d(layer)
    else:
      weights = layer.get_weights()
      if weights:
        cloned_layer.set_weights(weights)

  return quantized_model


def summarize_model(model, quantization_config=None):
  """Collects per-layer TurboQuant summaries for supported kernels."""
  quantization_config = TurboQuantConfig.from_dict(quantization_config)
  summaries = []
  for layer in model.layers:
    kernel = _quantizable_kernel(layer)
    if kernel is None or not _should_quantize_layer(layer, quantization_config):
      continue
    encoding = quantize_tensor(kernel, quantization_config)
    summary = summarize_encoding(kernel, encoding)
    summary['layer_name'] = layer.name
    summary['layer_type'] = layer.__class__.__name__
    summary['kernel_shape'] = tuple(int(dim) for dim in kernel.shape)
    summaries.append(summary)
  return summaries
