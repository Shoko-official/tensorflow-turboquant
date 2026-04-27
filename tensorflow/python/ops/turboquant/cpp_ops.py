"""Experimental C++ kernels for TurboQuant critical paths."""

import os

import tensorflow as tf

_CPP_OP_ENV_VAR = 'TURBOQUANT_CPP_KERNELS'
_CPP_OP_POLICY_AUTO = 'auto'
_CPP_OP_POLICY_DISABLED = 'disabled'
_CPP_OP_POLICY_REQUIRED = 'required'
_CPP_OP_DISABLED_VALUES = frozenset(('0', 'false', 'off', 'disable', 'disabled'))
_CPP_OP_REQUIRED_VALUES = frozenset(('1', 'true', 'on', 'require', 'required'))

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


def _cpp_kernel_policy() -> str:
  value = os.environ.get(_CPP_OP_ENV_VAR, '').strip().lower()
  if not value:
    return _CPP_OP_POLICY_AUTO
  if value in _CPP_OP_DISABLED_VALUES:
    return _CPP_OP_POLICY_DISABLED
  if value in _CPP_OP_REQUIRED_VALUES:
    return _CPP_OP_POLICY_REQUIRED
  raise ValueError(
      f'Unsupported {_CPP_OP_ENV_VAR}={value!r}. '
      'Expected auto, disabled/off/0, or required/on/1.'
  )


def cpp_kernel_status() -> dict[str, object]:
  """Returns runtime status for the experimental TurboQuant C++ op path."""
  policy = _cpp_kernel_policy()
  return {
      'env_var': _CPP_OP_ENV_VAR,
      'policy': policy,
      'available': _CPP_OP_MODULE is not None,
      'enabled': policy != _CPP_OP_POLICY_DISABLED,
      'using_cpp': policy != _CPP_OP_POLICY_DISABLED and _CPP_OP_MODULE is not None,
      'load_error': None if _CPP_OP_LOAD_ERROR is None else repr(_CPP_OP_LOAD_ERROR),
  }


def has_cpp_kernels() -> bool:
  """Returns whether the C++ kernels are loadable and enabled."""
  status = cpp_kernel_status()
  return bool(status['using_cpp'])


def _require_cpp_module():
  status = cpp_kernel_status()
  if not status['enabled']:
    raise RuntimeError(
        'TurboQuant C++ kernels are explicitly disabled via '
        f'{_CPP_OP_ENV_VAR}.'
    )
  if _CPP_OP_MODULE is None:
    raise RuntimeError(
        'TurboQuant C++ kernels are unavailable. '
        f'load_error={_CPP_OP_LOAD_ERROR!r}'
    )
  return _CPP_OP_MODULE


def unpack_indices_cpp(packed, flat_size: int, num_bits: int):
  """Unpacks packed TurboQuant indices via the C++ CPU op."""
  cpp_module = _require_cpp_module()
  packed_tensor = tf.convert_to_tensor(packed, dtype=tf.uint8)
  flat_size_tensor = tf.convert_to_tensor(flat_size, dtype=tf.int32)
  num_bits_tensor = tf.convert_to_tensor(num_bits, dtype=tf.int32)
  return cpp_module.turbo_quant_unpack_indices(
      packed=packed_tensor,
      flat_size=flat_size_tensor,
      num_bits=num_bits_tensor,
  )


def unpack_indices_with_fallback(packed, shape: tuple[int, ...], num_bits: int):
  """Uses the C++ op when enabled, otherwise falls back to Python unpacking."""
  status = cpp_kernel_status()
  flat_size = 1
  for dim in shape:
    flat_size *= int(dim)
  if status['using_cpp']:
    return unpack_indices_cpp(packed, flat_size=flat_size, num_bits=num_bits)
  if status['policy'] == _CPP_OP_POLICY_REQUIRED:
    _require_cpp_module()

  from tensorflow.python.ops.turboquant.core import unpack_indices  # pylint: disable=g-import-not-at-top

  unpacked = unpack_indices(packed, shape=shape, num_bits=num_bits)
  return tf.convert_to_tensor(unpacked, dtype=tf.uint8)
