"""Core Tensor TurboQuant routines."""

from dataclasses import dataclass
import math

import numpy as np

from tensorflow.python.ops.turboquant.config import TurboQuantConfig

_EPSILON = 1e-8


@dataclass(frozen=True)
class TurboQuantEncoding:
  """Packed representation for a quantized tensor."""

  original_shape: tuple[int, ...]
  axis: int
  group_size: int
  num_bits: int
  row_count: int
  padded_row_count: int
  indices: np.ndarray
  codebooks: np.ndarray
  scales: np.ndarray
  residual: np.ndarray

  @property
  def num_groups(self) -> int:
    return self.scales.shape[1]

  @property
  def channel_count(self) -> int:
    return self.codebooks.shape[0]

  @property
  def levels(self) -> int:
    return self.codebooks.shape[1]

  @property
  def nonzero_residual_count(self) -> int:
    return int(np.count_nonzero(self.residual))


def _as_float_array(value: np.ndarray | list[float]) -> np.ndarray:
  array = np.asarray(value, dtype=np.float32)
  if array.ndim == 0:
    raise ValueError('TurboQuant expects tensors with rank >= 1.')
  return array


def _nearest_codebook_indices(
    values: np.ndarray, codebook: np.ndarray
) -> np.ndarray:
  distances = np.abs(values[..., np.newaxis] - codebook[np.newaxis, :])
  return np.argmin(distances, axis=-1).astype(np.int32)


def _fit_codebook(samples: np.ndarray, config: TurboQuantConfig) -> np.ndarray:
  levels = config.levels
  if samples.size == 0:
    return np.zeros((levels,), dtype=np.float32)

  samples = np.asarray(samples, dtype=np.float32).reshape(-1)
  if np.allclose(samples, samples[0]):
    centers = np.full((levels,), samples[0], dtype=np.float32)
    centers[np.argmin(np.abs(centers))] = 0.0
    return np.sort(centers)

  percentiles = np.linspace(0.0, 100.0, num=levels, dtype=np.float32)
  centers = np.percentile(samples, percentiles).astype(np.float32)
  unique = np.unique(samples)
  if unique.size < levels:
    centers[:unique.size] = unique
    centers[unique.size:] = unique[-1]

  for _ in range(config.max_iterations):
    assignments = _nearest_codebook_indices(samples, centers)
    new_centers = centers.copy()
    for cluster_id in range(levels):
      cluster_values = samples[assignments == cluster_id]
      if cluster_values.size:
        new_centers[cluster_id] = np.mean(cluster_values, dtype=np.float32)
    new_centers = np.sort(new_centers)
    if np.max(np.abs(new_centers - centers)) <= config.convergence_tolerance:
      centers = new_centers
      break
    centers = new_centers

  centers[np.argmin(np.abs(centers))] = 0.0
  return centers.astype(np.float32)


def _reshape_tensor(
    tensor: np.ndarray, config: TurboQuantConfig
) -> tuple[np.ndarray, int, int, int]:
  axis = config.canonical_axis(tensor.ndim)
  transposed = np.moveaxis(tensor, axis, -1)
  row_count = int(np.prod(transposed.shape[:-1]))
  channel_count = transposed.shape[-1]
  num_groups = int(math.ceil(row_count / config.group_size))
  padded_row_count = num_groups * config.group_size

  matrix = transposed.reshape(row_count, channel_count)
  if padded_row_count != row_count:
    pad_rows = padded_row_count - row_count
    matrix = np.pad(matrix, ((0, pad_rows), (0, 0)))

  grouped = matrix.reshape(
      num_groups, config.group_size, channel_count
  ).transpose(2, 0, 1)
  return grouped.astype(np.float32), axis, row_count, padded_row_count


def quantize_tensor(
    tensor: np.ndarray | list[float], config: TurboQuantConfig
) -> TurboQuantEncoding:
  """Quantizes a tensor using per-channel codebooks and block scales."""
  array = _as_float_array(tensor)
  grouped, axis, row_count, padded_row_count = _reshape_tensor(array, config)

  rms = np.sqrt(np.mean(np.square(grouped), axis=(1, 2), keepdims=True))
  if config.outlier_threshold > 0:
    outlier_mask = np.abs(grouped) > (
        np.maximum(rms, _EPSILON) * config.outlier_threshold
    )
  else:
    outlier_mask = np.zeros_like(grouped, dtype=bool)

  masked_grouped = np.where(outlier_mask, 0.0, grouped)
  scales = np.max(np.abs(masked_grouped), axis=2)
  scales = np.where(scales < _EPSILON, 1.0, scales).astype(np.float32)
  normalized = masked_grouped / scales[:, :, np.newaxis]

  channel_count, num_groups, group_size = normalized.shape
  codebooks = np.zeros((channel_count, config.levels), dtype=np.float32)
  indices = np.zeros((channel_count, num_groups, group_size), dtype=np.int32)
  quantized = np.zeros_like(grouped, dtype=np.float32)

  for channel_id in range(channel_count):
    channel_samples = normalized[channel_id][~outlier_mask[channel_id]]
    codebook = _fit_codebook(channel_samples, config)
    codebooks[channel_id] = codebook
    indices[channel_id] = _nearest_codebook_indices(
        normalized[channel_id], codebook
    )
    quantized[channel_id] = (
        codebook[indices[channel_id]] * scales[channel_id][:, np.newaxis]
    )

  residual = np.where(outlier_mask, grouped - quantized, 0.0).astype(np.float32)

  return TurboQuantEncoding(
      original_shape=array.shape,
      axis=axis,
      group_size=config.group_size,
      num_bits=config.num_bits,
      row_count=row_count,
      padded_row_count=padded_row_count,
      indices=indices,
      codebooks=codebooks,
      scales=scales,
      residual=residual,
  )


def dequantize_tensor(encoding: TurboQuantEncoding) -> np.ndarray:
  """Reconstructs a tensor from its TurboQuant encoding."""
  gathered = np.take_along_axis(
      encoding.codebooks[:, np.newaxis, :],
      encoding.indices,
      axis=2,
  )
  grouped = gathered * encoding.scales[:, :, np.newaxis] + encoding.residual
  grouped = grouped.transpose(1, 2, 0).reshape(
      encoding.padded_row_count, encoding.channel_count
  )
  grouped = grouped[: encoding.row_count]

  axis_shape = list(encoding.original_shape)
  channel_count = axis_shape.pop(encoding.axis)
  restored = grouped.reshape(axis_shape + [channel_count])
  return np.moveaxis(restored, -1, encoding.axis)


def estimate_packed_bytes(encoding: TurboQuantEncoding) -> int:
  """Returns the estimated packed footprint in bytes."""
  index_bits = encoding.indices.size * encoding.num_bits
  index_bytes = int(math.ceil(index_bits / 8.0))
  codebook_bytes = int(encoding.codebooks.size * encoding.codebooks.dtype.itemsize)
  scale_bytes = int(encoding.scales.size * encoding.scales.dtype.itemsize)
  residual_mask = np.abs(encoding.residual) > 0
  residual_values = int(np.count_nonzero(residual_mask))
  residual_bytes = residual_values * (
      encoding.residual.dtype.itemsize + np.dtype(np.int32).itemsize
  )
  return index_bytes + codebook_bytes + scale_bytes + residual_bytes


def original_bytes(tensor: np.ndarray | list[float]) -> int:
  array = _as_float_array(tensor)
  return int(array.size * array.dtype.itemsize)


def summarize_encoding(
    reference: np.ndarray | list[float],
    encoding: TurboQuantEncoding,
) -> dict[str, float]:
  """Returns error and compression statistics for an encoding."""
  reference_array = _as_float_array(reference)
  restored = dequantize_tensor(encoding)
  diff = reference_array - restored
  packed_bytes = estimate_packed_bytes(encoding)
  original_size = original_bytes(reference_array)
  return {
      'original_bytes': float(original_size),
      'packed_bytes': float(packed_bytes),
      'compression_ratio': (
          float(original_size) / float(packed_bytes) if packed_bytes else np.inf
      ),
      'mean_squared_error': float(np.mean(np.square(diff))),
      'max_abs_error': float(np.max(np.abs(diff))),
      'outlier_fraction': float(encoding.nonzero_residual_count)
      / float(reference_array.size),
  }
