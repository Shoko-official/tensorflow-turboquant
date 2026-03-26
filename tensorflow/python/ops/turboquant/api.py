"""High-level TurboQuant APIs."""

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import core as core_layers
from tensorflow.python.ops.turboquant.config import TurboQuantConfig
from tensorflow.python.ops.turboquant.core import quantize_tensor
from tensorflow.python.ops.turboquant.core import summarize_encoding
from tensorflow.python.ops.turboquant.keras import TurboDense


def _should_quantize_layer(layer, quantization_config: TurboQuantConfig) -> bool:
  weights = layer.get_weights()
  if (
      not isinstance(layer, core_layers.Dense)
      or not weights
      or not quantization_config.should_quantize(weights[0].size)
  ):
    return False

  encoding = quantize_tensor(weights[0], quantization_config)
  summary = summarize_encoding(weights[0], encoding)
  return summary['packed_bytes'] < summary['original_bytes']


def _clone_layer(layer, quantization_config: TurboQuantConfig):
  if _should_quantize_layer(layer, quantization_config):
    layer_config = layer.get_config()
    layer_config['quantization_config'] = quantization_config.to_dict()
    layer_config['trainable'] = False
    return TurboDense(**layer_config)
  return layer.__class__.from_config(layer.get_config())


def quantize_model(model, quantization_config=None):
  """Clones a Functional or Sequential model with TurboDense layers."""
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
    else:
      weights = layer.get_weights()
      if weights:
        cloned_layer.set_weights(weights)

  return quantized_model


def summarize_model(model, quantization_config=None):
  """Collects per-layer TurboQuant summaries for Dense kernels."""
  quantization_config = TurboQuantConfig.from_dict(quantization_config)
  summaries = []
  for layer in model.layers:
    if not isinstance(layer, core_layers.Dense):
      continue
    weights = layer.get_weights()
    if not weights or not _should_quantize_layer(layer, quantization_config):
      continue
    encoding = quantize_tensor(weights[0], quantization_config)
    summary = summarize_encoding(weights[0], encoding)
    summary['layer_name'] = layer.name
    summaries.append(summary)
  return summaries
