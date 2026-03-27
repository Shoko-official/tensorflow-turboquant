"""High-level TurboQuant APIs."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import convolutional as convolutional_layers
from tensorflow.python.keras.layers import core as core_layers
from tensorflow.python.ops.turboquant.config import TurboQuantConfig
from tensorflow.python.ops.turboquant.core import dequantize_tensor
from tensorflow.python.ops.turboquant.core import estimate_packed_bytes
from tensorflow.python.ops.turboquant.core import original_bytes
from tensorflow.python.ops.turboquant.core import quantize_tensor
from tensorflow.python.ops.turboquant.keras import TurboConv1D
from tensorflow.python.ops.turboquant.keras import TurboConv2D
from tensorflow.python.ops.turboquant.keras import TurboConv3D
from tensorflow.python.ops.turboquant.keras import TurboDense
from tensorflow.python.ops.turboquant.keras import TurboDepthwiseConv2D
from tensorflow.python.ops.turboquant.keras import TurboSeparableConv1D
from tensorflow.python.ops.turboquant.keras import TurboSeparableConv2D
from tensorflow.python.saved_model import load as saved_model_load
from tensorflow.python.saved_model import save as saved_model_save


_SUPPORTED_LAYER_TYPES = (
    core_layers.Dense,
    convolutional_layers.Conv1D,
    convolutional_layers.Conv2D,
    convolutional_layers.Conv3D,
    convolutional_layers.DepthwiseConv2D,
    convolutional_layers.SeparableConv1D,
    convolutional_layers.SeparableConv2D,
)

_QUANTIZED_LAYER_TYPES = (
    TurboDense,
    TurboConv1D,
    TurboConv2D,
    TurboConv3D,
    TurboDepthwiseConv2D,
    TurboSeparableConv1D,
    TurboSeparableConv2D,
)


def get_custom_objects():
  """Returns custom objects required to deserialize TurboQuant wrappers."""
  return {
      'TurboDense': TurboDense,
      'TurboConv1D': TurboConv1D,
      'TurboConv2D': TurboConv2D,
      'TurboConv3D': TurboConv3D,
      'TurboDepthwiseConv2D': TurboDepthwiseConv2D,
      'TurboSeparableConv1D': TurboSeparableConv1D,
      'TurboSeparableConv2D': TurboSeparableConv2D,
  }


def _is_quantized_model(model) -> bool:
  return any(isinstance(layer, _QUANTIZED_LAYER_TYPES) for layer in model.layers)


def _reshape_depthwise_kernel_for_quantization(kernel):
  row_count = int(np.prod(kernel.shape[:-2]))
  return kernel.reshape(row_count, kernel.shape[-2] * kernel.shape[-1])


def _kernel_component_summary(kernel_name, kernel, quantization_config):
  kernel = np.asarray(kernel, dtype=np.float32)
  if kernel_name == 'depthwise_kernel':
    reference = kernel
    encoding = quantize_tensor(
        _reshape_depthwise_kernel_for_quantization(kernel), quantization_config
    )
    restored = dequantize_tensor(encoding).reshape(kernel.shape)
  else:
    reference = kernel
    encoding = quantize_tensor(kernel, quantization_config)
    restored = dequantize_tensor(encoding)

  diff = reference - restored
  packed_bytes = float(estimate_packed_bytes(encoding))
  original_size = float(original_bytes(reference))
  element_count = int(reference.size)
  return {
      'kernel_name': kernel_name,
      'kernel_shape': tuple(int(dim) for dim in reference.shape),
      'element_count': element_count,
      'original_bytes': original_size,
      'packed_bytes': packed_bytes,
      'compression_ratio': (
          original_size / packed_bytes if packed_bytes else np.inf
      ),
      'sum_squared_error': float(np.sum(np.square(diff))),
      'max_abs_error': float(np.max(np.abs(diff))),
      'outlier_count': int(np.count_nonzero(np.abs(encoding.residual) > 0.0)),
  }


def _layer_kernel_components(layer):
  if isinstance(layer, TurboDense):
    return [('kernel', np.asarray(layer.dequantized_kernel()))]
  if isinstance(layer, (TurboConv1D, TurboConv2D, TurboConv3D)):
    return [('kernel', np.asarray(layer.dequantized_kernel()))]
  if isinstance(layer, TurboDepthwiseConv2D):
    return [('depthwise_kernel', np.asarray(layer.dequantized_kernel()))]
  if isinstance(layer, (TurboSeparableConv1D, TurboSeparableConv2D)):
    return [
        ('depthwise_kernel', np.asarray(layer.dequantized_depthwise_kernel())),
        ('pointwise_kernel', np.asarray(layer.dequantized_pointwise_kernel())),
    ]

  weights = layer.get_weights()
  if not weights:
    return []
  if isinstance(layer, _SUPPORTED_LAYER_TYPES):
    if isinstance(layer, (convolutional_layers.SeparableConv1D,
                          convolutional_layers.SeparableConv2D)):
      return [
          ('depthwise_kernel', weights[0]),
          ('pointwise_kernel', weights[1]),
      ]
    if isinstance(layer, convolutional_layers.DepthwiseConv2D):
      return [('depthwise_kernel', weights[0])]
    return [('kernel', weights[0])]
  return []


def _layer_quantization_summary(layer, quantization_config: TurboQuantConfig):
  """Analyzes whether a layer should be TurboQuant-quantized."""
  summary = {
      'layer_name': layer.name,
      'layer_type': layer.__class__.__name__,
      'status': 'skipped',
      'reason': 'unsupported_layer_type',
  }
  components = _layer_kernel_components(layer)
  if not components:
    return summary

  summary['kernel_count'] = len(components)
  total_elements = sum(int(np.asarray(kernel).size) for _, kernel in components)
  summary['num_elements'] = total_elements
  if len(components) == 1:
    summary['kernel_shape'] = tuple(int(dim) for dim in components[0][1].shape)
  else:
    summary['kernel_shapes'] = {
        kernel_name: tuple(int(dim) for dim in kernel.shape)
        for kernel_name, kernel in components
    }

  if any(
      not quantization_config.should_quantize(int(np.asarray(kernel).size))
      for _, kernel in components
  ):
    summary['reason'] = 'below_minimum_elements'
    return summary

  component_summaries = [
      _kernel_component_summary(kernel_name, kernel, quantization_config)
      for kernel_name, kernel in components
  ]
  total_original_bytes = sum(item['original_bytes'] for item in component_summaries)
  total_packed_bytes = sum(item['packed_bytes'] for item in component_summaries)
  total_squared_error = sum(
      item['sum_squared_error'] for item in component_summaries
  )
  total_outlier_count = sum(item['outlier_count'] for item in component_summaries)
  max_abs_error = max(item['max_abs_error'] for item in component_summaries)

  summary.update({
      'original_bytes': float(total_original_bytes),
      'packed_bytes': float(total_packed_bytes),
      'compression_ratio': (
          float(total_original_bytes) / float(total_packed_bytes)
          if total_packed_bytes else np.inf
      ),
      'mean_squared_error': float(total_squared_error) / float(total_elements),
      'max_abs_error': float(max_abs_error),
      'outlier_fraction': float(total_outlier_count) / float(total_elements),
  })
  if len(component_summaries) > 1:
    summary['kernel_components'] = [
        {
            'kernel_name': item['kernel_name'],
            'kernel_shape': item['kernel_shape'],
            'compression_ratio': item['compression_ratio'],
            'mean_squared_error': (
                item['sum_squared_error'] / float(item['element_count'])
            ),
            'max_abs_error': item['max_abs_error'],
            'outlier_fraction': (
                float(item['outlier_count']) / float(item['element_count'])
            ),
        }
        for item in component_summaries
    ]

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

  layer_config = layer.get_config()
  layer_config['quantization_config'] = quantization_config.to_dict()
  layer_config['trainable'] = False

  if isinstance(layer, core_layers.Dense):
    return TurboDense(**layer_config)
  if isinstance(layer, convolutional_layers.DepthwiseConv2D):
    layer_config.pop('filters', None)
    return TurboDepthwiseConv2D(**layer_config)
  if isinstance(layer, convolutional_layers.SeparableConv1D):
    return TurboSeparableConv1D(**layer_config)
  if isinstance(layer, convolutional_layers.SeparableConv2D):
    return TurboSeparableConv2D(**layer_config)
  if isinstance(layer, convolutional_layers.Conv1D):
    return TurboConv1D(**layer_config)
  if isinstance(layer, convolutional_layers.Conv2D):
    return TurboConv2D(**layer_config)
  if isinstance(layer, convolutional_layers.Conv3D):
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

  quantized_layer_types = (
      TurboConv1D,
      TurboConv2D,
      TurboConv3D,
      TurboDepthwiseConv2D,
      TurboSeparableConv1D,
      TurboSeparableConv2D,
  )
  for layer in model.layers:
    cloned_layer = quantized_model.get_layer(layer.name)
    if isinstance(cloned_layer, TurboDense):
      cloned_layer.quantize_from_dense(layer)
    elif isinstance(cloned_layer, quantized_layer_types):
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
