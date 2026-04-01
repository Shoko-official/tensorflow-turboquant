"""High-level TurboQuant APIs."""

import itertools
import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import convolutional as convolutional_layers
from tensorflow.python.keras.layers import core as core_layers
from tensorflow.python.keras.layers import embeddings as embeddings_layers
from tensorflow.python.ops.turboquant.calibration import collect_calibration_stats
from tensorflow.python.ops.turboquant.config import CalibrationConfig
from tensorflow.python.ops.turboquant.config import TurboQuantConfig
from tensorflow.python.ops.turboquant.core import dequantize_tensor
from tensorflow.python.ops.turboquant.core import estimate_packed_bytes
from tensorflow.python.ops.turboquant.core import original_bytes
from tensorflow.python.ops.turboquant.core import quantize_tensor
from tensorflow.python.ops.turboquant.keras import TurboConv1D
from tensorflow.python.ops.turboquant.keras import TurboConv1DTranspose
from tensorflow.python.ops.turboquant.keras import TurboConv2D
from tensorflow.python.ops.turboquant.keras import TurboConv2DTranspose
from tensorflow.python.ops.turboquant.keras import TurboConv3D
from tensorflow.python.ops.turboquant.keras import TurboConv3DTranspose
from tensorflow.python.ops.turboquant.keras import TurboDense
from tensorflow.python.ops.turboquant.keras import TurboDepthwiseConv2D
from tensorflow.python.ops.turboquant.keras import TurboEmbedding
from tensorflow.python.ops.turboquant.keras import TurboSeparableConv1D
from tensorflow.python.ops.turboquant.keras import TurboSeparableConv2D
from tensorflow.python.saved_model import load as saved_model_load
from tensorflow.python.saved_model import save as saved_model_save


_SUPPORTED_LAYER_TYPES = (
    embeddings_layers.Embedding,
    core_layers.Dense,
    convolutional_layers.Conv1D,
    convolutional_layers.Conv1DTranspose,
    convolutional_layers.Conv2D,
    convolutional_layers.Conv2DTranspose,
    convolutional_layers.Conv3D,
    convolutional_layers.Conv3DTranspose,
    convolutional_layers.DepthwiseConv2D,
    convolutional_layers.SeparableConv1D,
    convolutional_layers.SeparableConv2D,
)

_QUANTIZED_LAYER_TYPES = (
    TurboEmbedding,
    TurboDense,
    TurboConv1D,
    TurboConv1DTranspose,
    TurboConv2D,
    TurboConv2DTranspose,
    TurboConv3D,
    TurboConv3DTranspose,
    TurboDepthwiseConv2D,
    TurboSeparableConv1D,
    TurboSeparableConv2D,
)


def get_custom_objects():
  """Returns custom objects required to deserialize TurboQuant wrappers."""
  return {
      'TurboEmbedding': TurboEmbedding,
      'TurboDense': TurboDense,
      'TurboConv1D': TurboConv1D,
      'TurboConv1DTranspose': TurboConv1DTranspose,
      'TurboConv2D': TurboConv2D,
      'TurboConv2DTranspose': TurboConv2DTranspose,
      'TurboConv3D': TurboConv3D,
      'TurboConv3DTranspose': TurboConv3DTranspose,
      'TurboDepthwiseConv2D': TurboDepthwiseConv2D,
      'TurboSeparableConv1D': TurboSeparableConv1D,
      'TurboSeparableConv2D': TurboSeparableConv2D,
  }


def _normalize_layer_name_set(layer_names):
  if layer_names is None:
    return None
  return {str(layer_name) for layer_name in layer_names}


def _validate_layer_name_selection(model,
                                   target_layer_names=None,
                                   exclude_layer_names=None):
  available = {layer.name for layer in model.layers}
  missing = set()
  for names in (target_layer_names, exclude_layer_names):
    if names is None:
      continue
    missing.update(set(names) - available)
  if missing:
    raise ValueError(
        'Unknown layer names in selection: ' + ', '.join(sorted(missing))
    )
  if target_layer_names is not None and exclude_layer_names is not None:
    overlap = set(target_layer_names).intersection(set(exclude_layer_names))
    if overlap:
      raise ValueError(
          'Layer names cannot be both targeted and excluded: '
          + ', '.join(sorted(overlap))
      )


def _is_layer_selected(layer_name, target_layer_names, exclude_layer_names):
  if target_layer_names is not None and layer_name not in target_layer_names:
    return False
  if exclude_layer_names is not None and layer_name in exclude_layer_names:
    return False
  return True


def _is_quantized_model(model) -> bool:
  return any(isinstance(layer, _QUANTIZED_LAYER_TYPES) for layer in model.layers)


def _reshape_depthwise_kernel_for_quantization(kernel):
  row_count = int(np.prod(kernel.shape[:-2]))
  return kernel.reshape(row_count, kernel.shape[-2] * kernel.shape[-1])


def _decision_trace_base(quantization_config: TurboQuantConfig):
  return {
      'minimum_elements': int(quantization_config.minimum_elements),
      'max_normalized_mean_squared_error': (
          None
          if quantization_config.max_normalized_mean_squared_error is None
          else float(quantization_config.max_normalized_mean_squared_error)
      ),
      'max_normalized_max_abs_error': (
          None
          if quantization_config.max_normalized_max_abs_error is None
          else float(quantization_config.max_normalized_max_abs_error)
      ),
  }


def _set_final_reason(summary, reason):
  summary['reason'] = reason
  summary['decision_trace']['final_reason'] = reason


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
  if isinstance(layer, TurboEmbedding):
    return [('embeddings', np.asarray(layer.dequantized_embeddings()))]
  if isinstance(layer, TurboDense):
    return [('kernel', np.asarray(layer.dequantized_kernel()))]
  if isinstance(
      layer,
      (
          TurboConv1D,
          TurboConv1DTranspose,
          TurboConv2D,
          TurboConv2DTranspose,
          TurboConv3D,
          TurboConv3DTranspose,
      ),
  ):
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
    if isinstance(layer, embeddings_layers.Embedding):
      return [('embeddings', weights[0])]
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
      'decision_trace': _decision_trace_base(quantization_config),
  }
  components = _layer_kernel_components(layer)
  if not components:
    _set_final_reason(summary, 'unsupported_layer_type')
    return summary

  summary['kernel_count'] = len(components)
  total_elements = sum(int(np.asarray(kernel).size) for _, kernel in components)
  summary['num_elements'] = total_elements
  summary['decision_trace']['kernel_count'] = int(len(components))
  summary['decision_trace']['num_elements'] = int(total_elements)
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
    summary['decision_trace']['below_minimum_elements'] = True
    _set_final_reason(summary, 'below_minimum_elements')
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

  summary['decision_trace']['packed_profitable'] = (
      summary['packed_bytes'] < summary['original_bytes']
  )
  if summary['packed_bytes'] >= summary['original_bytes']:
    _set_final_reason(summary, 'packing_not_profitable')
    return summary

  summary['status'] = 'quantized'
  _set_final_reason(summary, 'quantized')
  return summary


def _merge_calibration_stats(summary, calibration_stats):
  if not calibration_stats:
    return summary
  stats = calibration_stats.get(summary['layer_name'])
  if not stats:
    return summary

  output_rms = max(float(stats['output_rms']), 1e-8)
  output_abs_max = max(float(stats['output_abs_max']), 1e-8)
  summary['calibration'] = dict(stats)
  summary['decision_trace']['has_calibration'] = True
  if 'mean_squared_error' in summary:
    summary['normalized_mean_squared_error'] = (
        float(summary['mean_squared_error']) / float(output_rms * output_rms)
    )
    summary['normalized_max_abs_error'] = (
        float(summary['max_abs_error']) / float(output_abs_max)
    )
  return summary


def _apply_activation_guidance(summary, quantization_config: TurboQuantConfig):
  if summary.get('status') != 'quantized':
    return summary
  if not quantization_config.uses_activation_guidance:
    summary['decision_trace']['activation_guidance_enabled'] = False
    return summary
  summary['decision_trace']['activation_guidance_enabled'] = True
  if 'normalized_mean_squared_error' not in summary:
    return summary

  if (
      quantization_config.max_normalized_mean_squared_error is not None
      and summary['normalized_mean_squared_error']
      > quantization_config.max_normalized_mean_squared_error
  ):
    summary['status'] = 'skipped'
    _set_final_reason(summary, 'normalized_mean_squared_error_too_high')
    return summary

  if (
      quantization_config.max_normalized_max_abs_error is not None
      and summary['normalized_max_abs_error']
      > quantization_config.max_normalized_max_abs_error
  ):
    summary['status'] = 'skipped'
    _set_final_reason(summary, 'normalized_max_abs_error_too_high')
    return summary
  return summary


def _resolve_calibration_stats(model,
                               quantization_config: TurboQuantConfig,
                               calibration_stats=None,
                               representative_dataset=None,
                               calibration_config=None):
  del quantization_config
  if calibration_stats is not None:
    return calibration_stats
  if representative_dataset is None:
    return None
  calibration_config = CalibrationConfig.from_dict(calibration_config)
  return collect_calibration_stats(
      model, representative_dataset, calibration_config
  )


def _clone_quantized_layer(layer, layer_quantization_config=None):
  if layer_quantization_config is None:
    return layer.__class__.from_config(layer.get_config())

  layer_config = layer.get_config()
  layer_config['quantization_config'] = layer_quantization_config.to_dict()
  layer_config['trainable'] = False

  if isinstance(layer, embeddings_layers.Embedding):
    return TurboEmbedding(**layer_config)
  if isinstance(layer, core_layers.Dense):
    return TurboDense(**layer_config)
  if isinstance(layer, convolutional_layers.Conv1DTranspose):
    return TurboConv1DTranspose(**layer_config)
  if isinstance(layer, convolutional_layers.Conv2DTranspose):
    return TurboConv2DTranspose(**layer_config)
  if isinstance(layer, convolutional_layers.Conv3DTranspose):
    return TurboConv3DTranspose(**layer_config)
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


def _normalize_layer_overrides(layer_quantization_overrides):
  if not layer_quantization_overrides:
    return {}
  normalized = {}
  for layer_name, layer_config in layer_quantization_overrides.items():
    if layer_config is None:
      normalized[layer_name] = None
      continue
    normalized[layer_name] = TurboQuantConfig.from_dict(layer_config)
  return normalized


def _constraint_satisfied(summary,
                          target_normalized_mean_squared_error,
                          target_normalized_max_abs_error,
                          target_compression_ratio):
  if summary.get('status') != 'quantized':
    return False
  if target_compression_ratio is not None:
    if summary.get('compression_ratio', 0.0) < float(target_compression_ratio):
      return False
  if target_normalized_mean_squared_error is not None:
    if 'normalized_mean_squared_error' not in summary:
      return False
    if summary['normalized_mean_squared_error'] > float(
        target_normalized_mean_squared_error
    ):
      return False
  if target_normalized_max_abs_error is not None:
    if 'normalized_max_abs_error' not in summary:
      return False
    if summary['normalized_max_abs_error'] > float(
        target_normalized_max_abs_error
    ):
      return False
  return True


def _build_candidate_config(base_config,
                            num_bits: int,
                            group_size: int,
                            outlier_threshold: float):
  return TurboQuantConfig(
      num_bits=int(num_bits),
      group_size=int(group_size),
      axis=base_config.axis,
      outlier_threshold=float(outlier_threshold),
      max_iterations=base_config.max_iterations,
      convergence_tolerance=base_config.convergence_tolerance,
      minimum_elements=base_config.minimum_elements,
      max_normalized_mean_squared_error=(
          base_config.max_normalized_mean_squared_error
      ),
      max_normalized_max_abs_error=(
          base_config.max_normalized_max_abs_error
      ),
  )


def _effective_group_candidates(group_candidates, layer_summary):
  num_elements = int(layer_summary.get('num_elements', 0))
  if num_elements <= 0:
    return group_candidates
  valid = [group_size for group_size in group_candidates if group_size <= num_elements]
  if not valid:
    valid = [min(group_candidates)]
  return tuple(sorted(set(valid)))


def recommend_layer_configs(
    model,
    quantization_config=None,
    calibration_stats=None,
    representative_dataset=None,
    calibration_config=None,
    target_normalized_mean_squared_error=None,
    target_normalized_max_abs_error=None,
    target_compression_ratio=None,
    candidate_num_bits=None,
    candidate_group_sizes=None,
    candidate_outlier_thresholds=None,
    target_layer_names=None,
    exclude_layer_names=None,
    include_skipped=True,
):
  """Searches a per-layer quantization configuration under optional targets."""
  base_config = TurboQuantConfig.from_dict(quantization_config)
  calibration_stats = _resolve_calibration_stats(
      model,
      base_config,
      calibration_stats=calibration_stats,
      representative_dataset=representative_dataset,
      calibration_config=calibration_config,
  )

  bits_candidates = tuple(candidate_num_bits or (2, 3, 4, base_config.num_bits))
  group_candidates = tuple(
      candidate_group_sizes or (8, 16, 32, 64, base_config.group_size)
  )
  outlier_candidates = tuple(
      candidate_outlier_thresholds
      or (4.0, 6.0, 8.0, base_config.outlier_threshold)
  )
  bits_candidates = tuple(sorted(set(int(v) for v in bits_candidates)))
  group_candidates = tuple(sorted(set(int(v) for v in group_candidates)))
  outlier_candidates = tuple(sorted(set(float(v) for v in outlier_candidates)))
  target_layer_names = _normalize_layer_name_set(target_layer_names)
  exclude_layer_names = _normalize_layer_name_set(exclude_layer_names)
  _validate_layer_name_selection(
      model,
      target_layer_names=target_layer_names,
      exclude_layer_names=exclude_layer_names,
  )

  recommendations = []
  for layer in model.layers:
    if not _is_layer_selected(
        layer.name, target_layer_names, exclude_layer_names
    ):
      if include_skipped:
        summary = _layer_quantization_summary(layer, base_config)
        summary['status'] = 'skipped'
        _set_final_reason(summary, 'not_selected_by_filter')
        recommendations.append(summary)
      continue

    base_summary = _layer_quantization_summary(layer, base_config)
    if 'kernel_count' not in base_summary:
      if include_skipped:
        recommendations.append(base_summary)
      continue

    layer_group_candidates = _effective_group_candidates(
        group_candidates, base_summary
    )
    candidate_search_space = (
        len(bits_candidates)
        * len(layer_group_candidates)
        * len(outlier_candidates)
    )
    candidate_summaries = []
    for num_bits, group_size, outlier_threshold in itertools.product(
        bits_candidates, layer_group_candidates, outlier_candidates
    ):
      try:
        candidate_config = _build_candidate_config(
            base_config,
            num_bits=num_bits,
            group_size=group_size,
            outlier_threshold=outlier_threshold,
        )
      except ValueError:
        continue
      summary = _layer_quantization_summary(layer, candidate_config)
      summary = _merge_calibration_stats(summary, calibration_stats)
      summary = _apply_activation_guidance(summary, candidate_config)
      summary['evaluated_quantization_config'] = candidate_config.to_dict()
      candidate_summaries.append(summary)

    valid_candidates = [
        candidate for candidate in candidate_summaries
        if _constraint_satisfied(
            candidate,
            target_normalized_mean_squared_error,
            target_normalized_max_abs_error,
            target_compression_ratio,
        )
    ]

    if valid_candidates:
      best = max(
          valid_candidates,
          key=lambda item: (
              float(item.get('compression_ratio', 0.0)),
              -float(
                  item.get(
                      'normalized_mean_squared_error',
                      item.get('mean_squared_error', 0.0),
                  )
              ),
          ),
      )
      recommendation = dict(best)
      recommendation['recommended_quantization_config'] = dict(
          best['evaluated_quantization_config']
      )
      recommendation['candidate_count'] = len(candidate_summaries)
      recommendation['candidate_search_space'] = int(candidate_search_space)
      _set_final_reason(recommendation, recommendation['reason'])
    else:
      quantized_candidates = [
          candidate for candidate in candidate_summaries
          if candidate.get('status') == 'quantized'
      ]
      if quantized_candidates:
        fallback = min(
            quantized_candidates,
            key=lambda item: float(
                item.get(
                    'normalized_mean_squared_error',
                    item.get('mean_squared_error', np.inf),
                )
            ),
        )
      elif candidate_summaries:
        fallback = candidate_summaries[0]
      else:
        fallback = base_summary

      recommendation = dict(fallback)
      recommendation['status'] = 'skipped'
      _set_final_reason(recommendation, 'no_candidate_meets_constraints')
      recommendation['candidate_count'] = len(candidate_summaries)
      recommendation['candidate_search_space'] = int(candidate_search_space)

    if recommendation['status'] != 'quantized' and not include_skipped:
      continue
    recommendations.append(recommendation)
  return recommendations


def _validate_strict_quantization(summaries, target_layer_names=None):
  if target_layer_names:
    names = list(target_layer_names)
    by_name = {summary['layer_name']: summary for summary in summaries}
    missing = [name for name in names if name not in by_name]
    if missing:
      raise ValueError(
          'Unknown `target_layer_names` entries: ' + ', '.join(sorted(missing))
      )
    failures = [
        by_name[name] for name in names if by_name[name]['status'] != 'quantized'
    ]
  else:
    failures = [
        summary for summary in summaries
        if summary.get('kernel_count', 0) > 0 and summary['status'] != 'quantized'
    ]

  if failures:
    details = ', '.join(
        f"{item['layer_name']}({item['reason']})" for item in failures
    )
    raise ValueError(
        '`strict=True` requires full quantization for the selected layers. '
        f'Failed layers: {details}'
    )


def _aggregate_summaries(summaries):
  aggregate = {
      'total_layers': int(len(summaries)),
      'supported_layers': 0,
      'quantized_layers': 0,
      'skipped_layers': 0,
      'total_original_bytes': 0.0,
      'total_packed_bytes': 0.0,
      'effective_compression_ratio': 0.0,
      'mean_squared_error': 0.0,
      'max_abs_error': 0.0,
      'outlier_fraction': 0.0,
      'skipped_reasons': {},
      'quantized_layer_names': [],
      'skipped_layer_names': [],
  }

  total_elements = 0
  weighted_squared_error = 0.0
  weighted_outliers = 0.0
  for summary in summaries:
    if 'kernel_count' not in summary:
      continue
    aggregate['supported_layers'] += 1
    if summary.get('status') == 'quantized':
      aggregate['quantized_layers'] += 1
      aggregate['quantized_layer_names'].append(summary['layer_name'])
      aggregate['total_original_bytes'] += float(summary.get('original_bytes', 0.0))
      aggregate['total_packed_bytes'] += float(summary.get('packed_bytes', 0.0))
      num_elements = int(summary.get('num_elements', 0))
      total_elements += num_elements
      weighted_squared_error += (
          float(summary.get('mean_squared_error', 0.0)) * float(num_elements)
      )
      weighted_outliers += (
          float(summary.get('outlier_fraction', 0.0)) * float(num_elements)
      )
      aggregate['max_abs_error'] = max(
          aggregate['max_abs_error'], float(summary.get('max_abs_error', 0.0))
      )
    else:
      aggregate['skipped_layers'] += 1
      aggregate['skipped_layer_names'].append(summary['layer_name'])
      reason = summary.get('reason', 'unknown')
      aggregate['skipped_reasons'][reason] = (
          int(aggregate['skipped_reasons'].get(reason, 0)) + 1
      )

  if aggregate['total_packed_bytes'] > 0:
    aggregate['effective_compression_ratio'] = (
        float(aggregate['total_original_bytes'])
        / float(aggregate['total_packed_bytes'])
    )
  if total_elements > 0:
    aggregate['mean_squared_error'] = (
        float(weighted_squared_error) / float(total_elements)
    )
    aggregate['outlier_fraction'] = (
        float(weighted_outliers) / float(total_elements)
    )
  return aggregate


def _collect_layer_summaries(model,
                             base_config: TurboQuantConfig,
                             calibration_stats,
                             layer_quantization_overrides=None,
                             auto_tune=False,
                             target_normalized_mean_squared_error=None,
                             target_normalized_max_abs_error=None,
                             target_compression_ratio=None,
                             candidate_num_bits=None,
                             candidate_group_sizes=None,
                             candidate_outlier_thresholds=None,
                             target_layer_names=None,
                             exclude_layer_names=None):
  layer_overrides = _normalize_layer_overrides(layer_quantization_overrides)
  auto_recommendations = None
  auto_overrides = {}
  auto_skip_layers = set()
  if auto_tune:
    auto_recommendations = recommend_layer_configs(
        model,
        quantization_config=base_config,
        calibration_stats=calibration_stats,
        target_normalized_mean_squared_error=target_normalized_mean_squared_error,
        target_normalized_max_abs_error=target_normalized_max_abs_error,
        target_compression_ratio=target_compression_ratio,
        candidate_num_bits=candidate_num_bits,
        candidate_group_sizes=candidate_group_sizes,
        candidate_outlier_thresholds=candidate_outlier_thresholds,
        target_layer_names=target_layer_names,
        exclude_layer_names=exclude_layer_names,
        include_skipped=True,
    )
    for recommendation in auto_recommendations:
      layer_name = recommendation['layer_name']
      if recommendation.get('recommended_quantization_config'):
        auto_overrides[layer_name] = TurboQuantConfig.from_dict(
            recommendation['recommended_quantization_config']
        )
      elif recommendation.get('kernel_count', 0) > 0:
        auto_skip_layers.add(layer_name)

  merged_overrides = dict(auto_overrides)
  merged_overrides.update(layer_overrides)
  auto_skip_layers -= set(merged_overrides.keys())
  recommendation_by_name = {
      item['layer_name']: item for item in (auto_recommendations or [])
  }

  summaries = []
  for layer in model.layers:
    if not _is_layer_selected(
        layer.name, target_layer_names, exclude_layer_names
    ):
      summary = _layer_quantization_summary(layer, base_config)
      summary['status'] = 'skipped'
      _set_final_reason(summary, 'not_selected_by_filter')
      summaries.append(summary)
      continue

    if layer.name in auto_skip_layers:
      summary = dict(recommendation_by_name[layer.name])
    else:
      layer_config = merged_overrides.get(layer.name, base_config)
      if layer_config is None:
        summary = _layer_quantization_summary(layer, base_config)
        summary['status'] = 'skipped'
        _set_final_reason(summary, 'forced_skip_layer_override')
      else:
        summary = _layer_quantization_summary(layer, layer_config)
        summary = _merge_calibration_stats(summary, calibration_stats)
        summary = _apply_activation_guidance(summary, layer_config)
        summary['applied_quantization_config'] = layer_config.to_dict()

      if auto_tune and layer.name in recommendation_by_name:
        recommendation = recommendation_by_name[layer.name]
        if recommendation.get('recommended_quantization_config'):
          summary['recommended_quantization_config'] = dict(
              recommendation['recommended_quantization_config']
          )
        summary['candidate_count'] = recommendation.get('candidate_count')

    summaries.append(summary)
  return summaries


def quantize_model(model,
                   quantization_config=None,
                   calibration_stats=None,
                   representative_dataset=None,
                   calibration_config=None,
                   layer_quantization_overrides=None,
                   auto_tune=False,
                   target_normalized_mean_squared_error=None,
                   target_normalized_max_abs_error=None,
                   target_compression_ratio=None,
                   candidate_num_bits=None,
                   candidate_group_sizes=None,
                   candidate_outlier_thresholds=None,
                   dry_run=False,
                   strict=False,
                   target_layer_names=None,
                   exclude_layer_names=None,
                   return_report=False):
  """Clones a Functional or Sequential model with TurboQuant wrappers."""
  base_config = TurboQuantConfig.from_dict(quantization_config)
  if not hasattr(model, 'layers') or not hasattr(model, 'get_layer'):
    raise ValueError(
        '`quantize_model` only supports Keras models with an explicit layer '
        'graph.'
    )
  if not model.built:
    raise ValueError('`quantize_model` expects a built model.')
  target_layer_names = _normalize_layer_name_set(target_layer_names)
  exclude_layer_names = _normalize_layer_name_set(exclude_layer_names)
  _validate_layer_name_selection(
      model,
      target_layer_names=target_layer_names,
      exclude_layer_names=exclude_layer_names,
  )

  calibration_stats = _resolve_calibration_stats(
      model,
      base_config,
      calibration_stats=calibration_stats,
      representative_dataset=representative_dataset,
      calibration_config=calibration_config,
  )
  summaries = _collect_layer_summaries(
      model,
      base_config,
      calibration_stats,
      layer_quantization_overrides=layer_quantization_overrides,
      auto_tune=auto_tune,
      target_normalized_mean_squared_error=target_normalized_mean_squared_error,
      target_normalized_max_abs_error=target_normalized_max_abs_error,
      target_compression_ratio=target_compression_ratio,
      candidate_num_bits=candidate_num_bits,
      candidate_group_sizes=candidate_group_sizes,
      candidate_outlier_thresholds=candidate_outlier_thresholds,
      target_layer_names=target_layer_names,
      exclude_layer_names=exclude_layer_names,
  )

  if strict:
    strict_target_layer_names = target_layer_names
    if strict_target_layer_names is None and exclude_layer_names is not None:
      strict_target_layer_names = [
          layer.name
          for layer in model.layers
          if layer.name not in exclude_layer_names
      ]
    _validate_strict_quantization(summaries, strict_target_layer_names)
  if dry_run:
    if return_report:
      return {
          'summaries': summaries,
          'aggregate': _aggregate_summaries(summaries),
      }
    return summaries

  quantized_layer_configs = {}
  for summary in summaries:
    if summary.get('status') != 'quantized':
      continue
    config_dict = (
        summary.get('applied_quantization_config')
        or summary.get('recommended_quantization_config')
    )
    if config_dict is None:
      config_dict = base_config.to_dict()
    quantized_layer_configs[summary['layer_name']] = TurboQuantConfig.from_dict(
        config_dict
    )

  quantized_model = models.clone_model(
      model,
      clone_function=lambda layer: _clone_quantized_layer(
          layer, quantized_layer_configs.get(layer.name)
      ))
  if not quantized_model.built:
    quantized_model.build(model.input_shape)

  conv_like_quantized_layer_types = (
      TurboConv1D,
      TurboConv1DTranspose,
      TurboConv2D,
      TurboConv2DTranspose,
      TurboConv3D,
      TurboConv3DTranspose,
      TurboDepthwiseConv2D,
      TurboSeparableConv1D,
      TurboSeparableConv2D,
  )
  for layer in model.layers:
    cloned_layer = quantized_model.get_layer(layer.name)
    if (
        isinstance(layer, _QUANTIZED_LAYER_TYPES)
        and isinstance(cloned_layer, _QUANTIZED_LAYER_TYPES)
        and layer.__class__ == cloned_layer.__class__
    ):
      source_weights = layer.get_weights()
      if source_weights:
        cloned_layer.set_weights(source_weights)
      continue

    if isinstance(cloned_layer, TurboEmbedding):
      cloned_layer.quantize_from_embedding(layer)
    elif isinstance(cloned_layer, TurboDense):
      cloned_layer.quantize_from_dense(layer)
    elif isinstance(cloned_layer, conv_like_quantized_layer_types):
      cloned_layer.quantize_from_conv(layer)
    else:
      weights = layer.get_weights()
      if weights:
        cloned_layer.set_weights(weights)

  if return_report:
    return {
        'model': quantized_model,
        'summaries': summaries,
        'aggregate': _aggregate_summaries(summaries),
    }
  return quantized_model


def summarize_model(model,
                    quantization_config=None,
                    include_skipped=False,
                    calibration_stats=None,
                    representative_dataset=None,
                    calibration_config=None,
                    layer_quantization_overrides=None,
                    auto_tune=False,
                    target_normalized_mean_squared_error=None,
                    target_normalized_max_abs_error=None,
                    target_compression_ratio=None,
                    candidate_num_bits=None,
                    candidate_group_sizes=None,
                    candidate_outlier_thresholds=None,
                    target_layer_names=None,
                    exclude_layer_names=None,
                    return_report=False):
  """Collects per-layer TurboQuant summaries for supported kernels."""
  base_config = TurboQuantConfig.from_dict(quantization_config)
  target_layer_names = _normalize_layer_name_set(target_layer_names)
  exclude_layer_names = _normalize_layer_name_set(exclude_layer_names)
  _validate_layer_name_selection(
      model,
      target_layer_names=target_layer_names,
      exclude_layer_names=exclude_layer_names,
  )
  calibration_stats = _resolve_calibration_stats(
      model,
      base_config,
      calibration_stats=calibration_stats,
      representative_dataset=representative_dataset,
      calibration_config=calibration_config,
  )
  summaries = _collect_layer_summaries(
      model,
      base_config,
      calibration_stats,
      layer_quantization_overrides=layer_quantization_overrides,
      auto_tune=auto_tune,
      target_normalized_mean_squared_error=target_normalized_mean_squared_error,
      target_normalized_max_abs_error=target_normalized_max_abs_error,
      target_compression_ratio=target_compression_ratio,
      candidate_num_bits=candidate_num_bits,
      candidate_group_sizes=candidate_group_sizes,
      candidate_outlier_thresholds=candidate_outlier_thresholds,
      target_layer_names=target_layer_names,
      exclude_layer_names=exclude_layer_names,
  )
  filtered = summaries
  if not include_skipped:
    filtered = [
        summary for summary in summaries if summary['status'] == 'quantized'
    ]
  if return_report:
    return {
        'summaries': filtered,
        'aggregate': _aggregate_summaries(summaries),
    }
  return filtered


def export_saved_model(model,
                       export_dir,
                       quantization_config=None,
                       calibration_stats=None,
                       representative_dataset=None,
                       calibration_config=None,
                       layer_quantization_overrides=None,
                       auto_tune=False,
                       target_normalized_mean_squared_error=None,
                       target_normalized_max_abs_error=None,
                       target_compression_ratio=None,
                       candidate_num_bits=None,
                       candidate_group_sizes=None,
                       candidate_outlier_thresholds=None,
                       strict=False,
                       target_layer_names=None,
                       exclude_layer_names=None,
                       signatures=None,
                       options=None,
                       signature_key='serving_default'):
  """Exports a TurboQuant model as a TensorFlow SavedModel."""
  quantized_model = (
      model
      if _is_quantized_model(model)
      else quantize_model(
          model,
          quantization_config,
          calibration_stats=calibration_stats,
          representative_dataset=representative_dataset,
          calibration_config=calibration_config,
          layer_quantization_overrides=layer_quantization_overrides,
          auto_tune=auto_tune,
          target_normalized_mean_squared_error=(
              target_normalized_mean_squared_error
          ),
          target_normalized_max_abs_error=target_normalized_max_abs_error,
          target_compression_ratio=target_compression_ratio,
          candidate_num_bits=candidate_num_bits,
          candidate_group_sizes=candidate_group_sizes,
          candidate_outlier_thresholds=candidate_outlier_thresholds,
          strict=strict,
          target_layer_names=target_layer_names,
          exclude_layer_names=exclude_layer_names,
      )
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
