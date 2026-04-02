"""Core Tensor TurboQuant routines."""

from dataclasses import dataclass
import math

import numpy as np

from tensorflow.python.ops.turboquant.config import TurboQuantConfig

_EPSILON = 1e-8
_ENCODING_FORMAT_NAME = 'turboquant_encoding'
_ENCODING_FORMAT_VERSION = 1


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
  if not np.all(np.isfinite(array)):
    raise ValueError('TurboQuant only supports finite tensor values.')
  return array


def _nearest_codebook_indices(
    values: np.ndarray, codebook: np.ndarray
) -> np.ndarray:
  codebook = np.asarray(codebook, dtype=np.float32).reshape(-1)
  if codebook.size == 1:
    return np.zeros(values.shape, dtype=np.int32)

  # Codebooks are sorted during fitting, so a binary-search projection avoids
  # building a large [values, levels] distance tensor.
  upper = np.searchsorted(codebook, values, side='left')
  upper = np.clip(upper, 0, codebook.size - 1)
  lower = np.clip(upper - 1, 0, codebook.size - 1)
  upper_dist = np.abs(values - codebook[upper])
  lower_dist = np.abs(values - codebook[lower])
  return np.where(lower_dist <= upper_dist, lower, upper).astype(np.int32)


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
    cluster_sums = np.bincount(
        assignments, weights=samples, minlength=levels
    ).astype(np.float32)
    cluster_counts = np.bincount(assignments, minlength=levels)
    new_centers = np.where(
        cluster_counts > 0,
        cluster_sums / np.maximum(cluster_counts, 1),
        centers,
    ).astype(np.float32)
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
  index_dtype = np.uint8 if config.levels <= np.iinfo(np.uint8).max + 1 else np.int32
  indices = np.zeros((channel_count, num_groups, group_size), dtype=index_dtype)
  quantized = np.zeros_like(grouped, dtype=np.float32)

  for channel_id in range(channel_count):
    channel_samples = normalized[channel_id][~outlier_mask[channel_id]]
    codebook = _fit_codebook(channel_samples, config)
    codebooks[channel_id] = codebook
    channel_indices = _nearest_codebook_indices(normalized[channel_id], codebook)
    indices[channel_id] = channel_indices.astype(index_dtype, copy=False)
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
  indices = encoding.indices.astype(np.intp, copy=False)
  gathered = np.take_along_axis(
      encoding.codebooks[:, np.newaxis, :],
      indices,
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


def pack_indices(indices: np.ndarray, num_bits: int) -> np.ndarray:
  """Packs quantized indices in a compact little-endian bitstream."""
  if num_bits < 1 or num_bits > 8:
    raise ValueError(f'`num_bits` must be in [1, 8]. Got: {num_bits}.')
  flat = np.asarray(indices, dtype=np.int32).reshape(-1)
  if flat.size == 0:
    return np.zeros((0,), dtype=np.uint8)
  max_value = (1 << num_bits) - 1
  if np.any(flat < 0) or np.any(flat > max_value):
    raise ValueError(
        'All packed indices must be in '
        f'[0, {max_value}] for num_bits={num_bits}.'
    )

  total_bits = int(flat.size * num_bits)
  packed = np.zeros((int(math.ceil(total_bits / 8.0)),), dtype=np.uint8)
  bit_offset = 0
  for value in flat:
    byte_index = bit_offset // 8
    bit_index = bit_offset % 8
    packed[byte_index] |= (int(value) << bit_index) & 0xFF
    overflow_bits = bit_index + num_bits - 8
    if overflow_bits > 0 and byte_index + 1 < packed.size:
      packed[byte_index + 1] |= (int(value) >> (num_bits - overflow_bits)) & 0xFF
    bit_offset += num_bits
  return packed


def unpack_indices(
    packed: np.ndarray, shape: tuple[int, ...], num_bits: int
) -> np.ndarray:
  """Unpacks quantized indices from a compact little-endian bitstream."""
  if num_bits < 1 or num_bits > 8:
    raise ValueError(f'`num_bits` must be in [1, 8]. Got: {num_bits}.')
  flat_size = int(np.prod(shape))
  if flat_size == 0:
    return np.zeros(shape, dtype=np.uint8)
  packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
  total_bits = flat_size * num_bits
  needed_bytes = int(math.ceil(total_bits / 8.0))
  if packed.size < needed_bytes:
    raise ValueError(
        'Packed indices payload is too small for the requested shape and '
        f'bit width: need {needed_bytes} bytes, got {packed.size}.'
    )

  max_value = (1 << num_bits) - 1
  flat = np.zeros((flat_size,), dtype=np.uint8)
  bit_offset = 0
  for i in range(flat_size):
    byte_index = bit_offset // 8
    bit_index = bit_offset % 8
    value = int(packed[byte_index]) >> bit_index
    overflow_bits = bit_index + num_bits - 8
    if overflow_bits > 0:
      value |= int(packed[byte_index + 1]) << (8 - bit_index)
    flat[i] = np.uint8(value & max_value)
    bit_offset += num_bits
  return flat.reshape(shape)


def serialize_encoding(encoding: TurboQuantEncoding) -> dict[str, object]:
  """Returns a serialized dictionary with packed TurboQuant indices."""
  return {
      'format': _ENCODING_FORMAT_NAME,
      'format_version': _ENCODING_FORMAT_VERSION,
      'original_shape': tuple(int(dim) for dim in encoding.original_shape),
      'axis': int(encoding.axis),
      'group_size': int(encoding.group_size),
      'num_bits': int(encoding.num_bits),
      'row_count': int(encoding.row_count),
      'padded_row_count': int(encoding.padded_row_count),
      'indices_shape': tuple(int(dim) for dim in encoding.indices.shape),
      'indices_packed': pack_indices(encoding.indices, encoding.num_bits),
      'codebooks': np.asarray(encoding.codebooks, dtype=np.float32),
      'scales': np.asarray(encoding.scales, dtype=np.float32),
      'residual': np.asarray(encoding.residual, dtype=np.float32),
  }


def deserialize_encoding(payload: dict[str, object]) -> TurboQuantEncoding:
  """Restores a `TurboQuantEncoding` produced by `serialize_encoding`."""
  format_name = payload.get('format', _ENCODING_FORMAT_NAME)
  if format_name != _ENCODING_FORMAT_NAME:
    raise ValueError(
        'Unknown TurboQuant encoding format: '
        f'{format_name!r}. Expected {_ENCODING_FORMAT_NAME!r}.'
    )
  format_version = int(payload.get('format_version', 0))
  if format_version not in (0, _ENCODING_FORMAT_VERSION):
    raise ValueError(
        f'Unsupported TurboQuant encoding format version: {format_version}.'
    )

  required_keys = (
      'original_shape',
      'axis',
      'group_size',
      'num_bits',
      'row_count',
      'padded_row_count',
      'indices_shape',
      'indices_packed',
      'codebooks',
      'scales',
      'residual',
  )
  missing = [key for key in required_keys if key not in payload]
  if missing:
    raise ValueError(
        'TurboQuant encoding payload is missing required fields: '
        + ', '.join(sorted(missing))
    )

  indices_shape = tuple(int(dim) for dim in payload['indices_shape'])
  num_bits = int(payload['num_bits'])
  indices = unpack_indices(payload['indices_packed'], indices_shape, num_bits)
  return TurboQuantEncoding(
      original_shape=tuple(int(dim) for dim in payload['original_shape']),
      axis=int(payload['axis']),
      group_size=int(payload['group_size']),
      num_bits=num_bits,
      row_count=int(payload['row_count']),
      padded_row_count=int(payload['padded_row_count']),
      indices=indices,
      codebooks=np.asarray(payload['codebooks'], dtype=np.float32),
      scales=np.asarray(payload['scales'], dtype=np.float32),
      residual=np.asarray(payload['residual'], dtype=np.float32),
  )


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
