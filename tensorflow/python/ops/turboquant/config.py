"""Configuration objects for TurboQuant."""

from dataclasses import asdict
from dataclasses import dataclass


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

  def to_dict(self) -> dict[str, object]:
    return asdict(self)

  @classmethod
  def from_dict(cls, config: dict[str, object] | None) -> 'TurboQuantConfig':
    if isinstance(config, cls):
      return config
    return cls(**config) if config else cls()
