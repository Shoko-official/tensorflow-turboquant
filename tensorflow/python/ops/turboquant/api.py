"""High-level TurboQuant APIs."""

from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import convolutional as convolutional_layers
from tensorflow.python.keras.layers import core as core_layers
from tensorflow.python.ops.turboquant.config import TurboQuantConfig
from tensorflow.python.ops.turboquant.core import quantize_tensor
from tensorflow.python.ops.turboquant.core import summarize_encoding
from tensorflow.python.ops.turboquant.keras import TurboConv1D
from tensorflow.python.ops.turboquant.keras import TurboConv2D
from tensorflow.python.ops.turboquant.keras import TurboConv3D
from tensorflow.python.ops.turboquant.keras import TurboDense
from tensorflow.python.saved_model import load as saved_model_load
from tensorflow.python.saved_model import save as saved_model_save


_SUPPORTED_LAYER_TYPES = (
    core_layers.Dense,
    convolutional_layers.Conv1D,
    convolutional_layers.Conv2D,
    convolutional_layers.Conv3D,
)

_QUANTIZED_LAYER_TYPES = (
    TurboDense,
    TurboConv1D,
    TurboConv2D,
    TurboConv3D,
)


def get_custom_objects():
  """Returns custom objects required to deserialize TurboQuant wrappers."""
  return {
      'TurboDense': TurboDense,
      'TurboConv1D': TurboConv1D,
      'TurboConv2D': TurboConv2D,
      'TurboConv3D': TurboConv3D,
  }


def _is_quantized_model(model) -> bool:
  return any(isinstance(layer, _QUANTIZED_LAYER_TYPES) for layer in model.layers)


def _quantizable_kernel(layer):
  weights = layer.get_weights()
  if (
      not isinstance(layer, _SUPPORTED_LAYER_TYPES + _QUANTIZED_LAYER_TYPES)
      or not weights
  ):
    return None
  return weights[0]


def _layer_quantization_summary(layer, quantization_config: TurboQuantConfig):
  """Analyzes whether a layer should be TurboQuant-quantized."""
  summary = {
      'layer_name': layer.name,
      'layer_type': layer.__class__.__name__,
      'status': 'skipped',
      'reason': 'unsupported_layer_type',
  }
  kernel = _quantizable_kernel(layer)
  if kernel is None:
    return summary

  summary['kernel_shape'] = tuple(int(dim) for dim in kernel.shape)
  summary['num_elements'] = int(kernel.size)
  if not quantization_config.should_quantize(kernel.size):
    summary['reason'] = 'below_minimum_elements'
    return summary

  encoding = quantize_tensor(kernel, quantization_config)
  summary.update(summarize_encoding(kernel, encoding))
  if summary['packed_bytes'] >= summary['original_bytes']:
    summary['reason'] = 'packing_not_profitable'
    return summary

  summary['status'] = 'quantized'
  summary['reason'] = 'quantized'
  return summary


def _should_quantize_layer(layer, quantization_config: TurboQuantConfig) -> bool:
  return (
      _layer_quantization_summary(layer, quantization_config)['status']
      == 'quantized'
  )


def _clone_layer(layer, quantization_config: TurboQuantConfig):
  if not _should_quantize_layer(layer, quantization_config):
    return layer.__class__.from_config(layer.get_config())

  if isinstance(layer, core_layers.Dense):
    layer_config = layer.get_config()
    layer_config['quantization_config'] = quantization_config.to_dict()
    layer_config['trainable'] = False
    return TurboDense(**layer_config)
  if isinstance(layer, convolutional_layers.Conv1D):
    layer_config = layer.get_config()
    layer_config['quantization_config'] = quantization_config.to_dict()
    layer_config['trainable'] = False
    return TurboConv1D(**layer_config)
  if isinstance(layer, convolutional_layers.Conv2D):
    layer_config = layer.get_config()
    layer_config['quantization_config'] = quantization_config.to_dict()
    layer_config['trainable'] = False
    return TurboConv2D(**layer_config)
  if isinstance(layer, convolutional_layers.Conv3D):
    layer_config = layer.get_config()
    layer_config['quantization_config'] = quantization_config.to_dict()
    layer_config['trainable'] = False
    return TurboConv3D(**layer_config)
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
    elif isinstance(cloned_layer, (TurboConv1D, TurboConv2D, TurboConv3D)):
      cloned_layer.quantize_from_conv(layer)
    else:
      weights = layer.get_weights()
      if weights:
        cloned_layer.set_weights(weights)

  return quantized_model


def summarize_model(model, quantization_config=None, include_skipped=False):
  """Collects per-layer TurboQuant summaries for supported kernels."""
  quantization_config = TurboQuantConfig.from_dict(quantization_config)
  summaries = []
  for layer in model.layers:
    summary = _layer_quantization_summary(layer, quantization_config)
    if summary['status'] != 'quantized' and not include_skipped:
      continue
    summaries.append(summary)
  return summaries


def export_saved_model(model,
                       export_dir,
                       quantization_config=None,
                       signatures=None,
                       options=None,
                       signature_key='serving_default'):
  """Exports a TurboQuant model as a TensorFlow SavedModel."""
  quantized_model = (
      model
      if _is_quantized_model(model)
      else quantize_model(model, quantization_config)
  )

  if signatures is None:
    input_signature = [
        tensor_spec.TensorSpec(
            shape=tensor.shape,
            dtype=tensor.dtype,
            name=tensor.name.split(':')[0],
        )
        for tensor in quantized_model.inputs
    ]

    @def_function.function(input_signature=input_signature)
    def serving_default(*args):
      model_inputs = args[0] if len(args) == 1 else list(args)
      outputs = quantized_model(model_inputs)
      if isinstance(outputs, dict):
        return outputs
      if isinstance(outputs, (list, tuple)):
        return {
            f'output_{index}': output for index, output in enumerate(outputs)
        }
      return {'outputs': outputs}

    signatures = {signature_key: serving_default}

  saved_model_save.save(
      quantized_model,
      export_dir,
      signatures=signatures,
      options=options,
  )
  return quantized_model


def load_saved_model(export_dir, tags=None, options=None):
  """Loads a TurboQuant SavedModel exported with `export_saved_model`."""
  return saved_model_load.load(export_dir, tags=tags, options=options)
