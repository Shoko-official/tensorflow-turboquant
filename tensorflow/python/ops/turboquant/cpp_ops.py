"""Experimental C++ kernels for TurboQuant critical paths."""

import os

import tensorflow as tf

_CPP_OP_MODULE = None
_CPP_OP_LOAD_ERROR = None

try:
  _CPP_OP_MODULE = tf.load_op_library(
      os.path.join(
          tf.compat.v1.resource_loader.get_data_files_path(),
          'turboquant_packed_indices_op.so',
      )
  )
except (tf.errors.NotFoundError, OSError) as error:  # pragma: no cover
  _CPP_OP_LOAD_ERROR = error


def has_cpp_kernels() -> bool:
  return _CPP_OP_MODULE is not None


def unpack_indices_cpp(packed, flat_size: int, num_bits: int):
  """Unpacks packed TurboQuant indices via the C++ CPU op."""
  if _CPP_OP_MODULE is None:
    raise RuntimeError(
        'TurboQuant C++ kernels are unavailable. '
        f'load_error={_CPP_OP_LOAD_ERROR!r}'
    )
  packed_tensor = tf.convert_to_tensor(packed, dtype=tf.uint8)
  flat_size_tensor = tf.convert_to_tensor(flat_size, dtype=tf.int32)
  num_bits_tensor = tf.convert_to_tensor(num_bits, dtype=tf.int32)
  return _CPP_OP_MODULE.turbo_quant_unpack_indices(
      packed=packed_tensor,
      flat_size=flat_size_tensor,
      num_bits=num_bits_tensor,
  )
