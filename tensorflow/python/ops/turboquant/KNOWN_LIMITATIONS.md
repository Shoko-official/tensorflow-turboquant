# TurboQuant Known Limitations

- Current quantization flow is inference-oriented and weight-only.
- C++ path currently accelerates packed-index unpack only; the rest of the
  quantization pipeline still runs in Python/NumPy/TensorFlow ops. Runtime
  selection now has an explicit fallback policy, but `pack` and dequant gather
  kernels are still pending.
- Benchmark scripts now expose task-level metrics and CI gates, but the default
  synthetic dataset smoke remains weaker than a representative downstream task
  evaluation.
- Real-dataset benchmark path expects local `.npz` inputs and does not bundle
  dataset download logic.
- Runtime gains depend on layer mix and deployment environment; small models may
  show minimal latency improvements.
