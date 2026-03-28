"""Configuration objects for TurboQuant."""

from dataclasses import asdict
from dataclasses import dataclass


def _strict_from_dict(cls, config):
  if isinstance(config, cls):
    return config
  if not config:
    return cls()

  unknown_keys = set(config.keys()) - set(cls.__dataclass_fields__.keys())
  if unknown_keys:
    raise ValueError(
        f'Unknown `{cls.__name__}` keys: {sorted(unknown_keys)}.'
    )
  return cls(**config)


@dataclass(frozen=True)
class TurboQuantConfig:
  """Configuration for block-wise codebook quantization."""

  num_bits: int = 4
  group_size: int = 64
  axis: int = -1
  outlier_threshold: float = 6.0
  max_iterations: int = 25
  convergence_tolerance: float = 1e-4
  minimum_elements: int = 256
  max_normalized_mean_squared_error: float | None = None
  max_normalized_max_abs_error: float | None = None

  def __post_init__(self):
    if self.num_bits < 1 or self.num_bits > 8:
      raise ValueError(
          f'`num_bits` must be in the [1, 8] range. Got: {self.num_bits}.'
      )
    if self.group_size < 1:
      raise ValueError(
          f'`group_size` must be a positive integer. Got: {self.group_size}.'
      )
    if self.outlier_threshold < 0:
      raise ValueError(
          '`outlier_threshold` must be non-negative. '
          f'Got: {self.outlier_threshold}.'
      )
    if self.max_iterations < 1:
      raise ValueError(
          '`max_iterations` must be a positive integer. '
          f'Got: {self.max_iterations}.'
      )
    if self.convergence_tolerance <= 0:
      raise ValueError(
          '`convergence_tolerance` must be strictly positive. '
          f'Got: {self.convergence_tolerance}.'
      )
    if self.minimum_elements < 1:
      raise ValueError(
          '`minimum_elements` must be a positive integer. '
          f'Got: {self.minimum_elements}.'
      )
    if (
        self.max_normalized_mean_squared_error is not None
        and self.max_normalized_mean_squared_error <= 0
    ):
      raise ValueError(
          '`max_normalized_mean_squared_error` must be strictly positive '
          f'when set. Got: {self.max_normalized_mean_squared_error}.'
      )
    if (
        self.max_normalized_max_abs_error is not None
        and self.max_normalized_max_abs_error <= 0
    ):
      raise ValueError(
          '`max_normalized_max_abs_error` must be strictly positive when '
          f'set. Got: {self.max_normalized_max_abs_error}.'
      )

  @property
  def levels(self) -> int:
    return 1 << self.num_bits

  def canonical_axis(self, rank: int) -> int:
    axis = self.axis if self.axis >= 0 else rank + self.axis
    if axis < 0 or axis >= rank:
      raise ValueError(
          f'`axis`={self.axis} is out of bounds for rank {rank}.'
      )
    return axis

  def should_quantize(self, num_elements: int) -> bool:
    return num_elements >= self.minimum_elements

  @property
  def uses_activation_guidance(self) -> bool:
    return (
        self.max_normalized_mean_squared_error is not None
        or self.max_normalized_max_abs_error is not None
    )

  def to_dict(self) -> dict[str, object]:
    return asdict(self)

  @classmethod
  def from_dict(cls, config: dict[str, object] | None) -> 'TurboQuantConfig':
    return _strict_from_dict(cls, config)


@dataclass(frozen=True)
class CalibrationConfig:
  """Configuration for representative-dataset calibration."""

  max_steps: int = 32
  max_samples: int = 4096

  def __post_init__(self):
    if self.max_steps < 1:
      raise ValueError(
          '`max_steps` must be a positive integer. '
          f'Got: {self.max_steps}.'
      )
    if self.max_samples < 1:
      raise ValueError(
          '`max_samples` must be a positive integer. '
          f'Got: {self.max_samples}.'
      )

  def to_dict(self) -> dict[str, object]:
    return asdict(self)

  @classmethod
  def from_dict(cls, config: dict[str, object] | None) -> 'CalibrationConfig':
    return _strict_from_dict(cls, config)
