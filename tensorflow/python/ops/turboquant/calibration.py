"""Representative-dataset calibration utilities for TurboQuant."""

import math

import numpy as np

from tensorflow.python.keras import models
from tensorflow.python.ops.turboquant.config import CalibrationConfig

_EPSILON = 1e-8


def _as_model_inputs(batch):
  if isinstance(batch, tuple):
    return batch[0]
  return batch


def _flatten_arrays(value):
  if isinstance(value, dict):
    arrays = []
    for item in value.values():
      arrays.extend(_flatten_arrays(item))
    return arrays
  if isinstance(value, (list, tuple)):
    arrays = []
    for item in value:
      arrays.extend(_flatten_arrays(item))
    return arrays
  return [np.asarray(value)]


def _infer_batch_size(model_inputs) -> int:
  arrays = _flatten_arrays(model_inputs)
  for array in arrays:
    if array.ndim:
      return int(array.shape[0])
  return 1


def _create_accumulator(layer):
  return {
      'layer_name': layer.name,
      'layer_type': layer.__class__.__name__,
      'sample_count': 0,
      'value_count': 0,
      'sum_abs': 0.0,
      'sum_sq': 0.0,
      'output_abs_max': 0.0,
  }


def _update_accumulator(accumulator, outputs, batch_size: int):
  arrays = _flatten_arrays(outputs)
  accumulator['sample_count'] += batch_size
  for array in arrays:
    if not np.issubdtype(array.dtype, np.number):
      continue
    values = np.asarray(array, dtype=np.float32)
    accumulator['value_count'] += int(values.size)
    accumulator['sum_abs'] += float(np.sum(np.abs(values)))
    accumulator['sum_sq'] += float(np.sum(np.square(values)))
    accumulator['output_abs_max'] = max(
        accumulator['output_abs_max'],
        float(np.max(np.abs(values))) if values.size else 0.0,
    )


def _finalize_accumulator(accumulator):
  value_count = accumulator.pop('value_count')
  sum_abs = accumulator.pop('sum_abs')
  sum_sq = accumulator.pop('sum_sq')
  if not value_count:
    accumulator['output_abs_mean'] = 0.0
    accumulator['output_rms'] = 0.0
    accumulator['dynamic_range'] = 0.0
    return accumulator

  output_abs_mean = sum_abs / float(value_count)
  output_rms = math.sqrt(sum_sq / float(value_count))
  accumulator['output_abs_mean'] = float(output_abs_mean)
  accumulator['output_rms'] = float(output_rms)
  accumulator['dynamic_range'] = float(
      accumulator['output_abs_max'] / max(output_rms, _EPSILON)
  )
  return accumulator


def collect_calibration_stats(model, representative_dataset, calibration_config=None):
  """Collects per-layer activation statistics from a representative dataset."""
  calibration_config = CalibrationConfig.from_dict(calibration_config)
  if not model.built:
    raise ValueError('`collect_calibration_stats` expects a built model.')
  if not getattr(model, 'inputs', None):
    raise ValueError(
        '`collect_calibration_stats` only supports graph-connected Keras models.'
    )

  probe_layers = [
      layer for layer in model.layers
      if getattr(layer, 'output', None) is not None
      and layer.__class__.__name__ != 'InputLayer'
  ]
  probe_model = models.Model(
      inputs=model.inputs,
      outputs=[layer.output for layer in probe_layers],
  )
  accumulators = {
      layer.name: _create_accumulator(layer) for layer in probe_layers
  }

  sample_count = 0
  for step, batch in enumerate(representative_dataset):
    if step >= calibration_config.max_steps:
      break
    model_inputs = _as_model_inputs(batch)
    batch_size = _infer_batch_size(model_inputs)
    outputs = probe_model(model_inputs, training=False)
    if not isinstance(outputs, (list, tuple)):
      outputs = [outputs]

    for layer, layer_outputs in zip(probe_layers, outputs):
      _update_accumulator(accumulators[layer.name], layer_outputs, batch_size)

    sample_count += batch_size
    if sample_count >= calibration_config.max_samples:
      break

  return {
      layer_name: _finalize_accumulator(accumulator)
      for layer_name, accumulator in accumulators.items()
  }
