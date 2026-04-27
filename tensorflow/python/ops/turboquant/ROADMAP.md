# TurboQuant Roadmap

## Near Term

- Stabilize benchmark protocol on fixed hardware profiles.
- Promote benchmark thresholds into CI regression gates and keep the seed/threshold
  matrix reproducible across local and CI runs.
- Expand layer support to additional normalization and attention blocks.
- Harden SavedModel interoperability tests for serialized TurboQuant payloads.

## Mid Term

- Promote packed-index C++ custom op from experimental to default path once the
  fallback policy, parity coverage, and end-to-end gains are proven.
- Add additional C++ kernels for packing and codebook gather hotspots.
- Evaluate integration with TensorFlow graph/XLA fusion paths.
- Publish reproducible benchmark artifacts for representative model families.

## Long Term

- Add hardware-specific kernels (AVX2/AVX512/NEON where relevant).
- Add optional training-aware calibration extensions.
- Align TurboQuant export metadata with broader TensorFlow quantization tooling.
