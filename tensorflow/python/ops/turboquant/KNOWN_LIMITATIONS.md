# TurboQuant Known Limitations

- Current quantization flow is inference-oriented and weight-only.
- C++ path currently accelerates packed-index unpack only; the rest of the
  quantization pipeline still runs in Python/NumPy/TensorFlow ops.
- Benchmark scripts focus on drift/compression/latency and do not guarantee
  task-level quality unless the input model is trained on representative data.
- Real-dataset benchmark path expects local `.npz` inputs and does not bundle
  dataset download logic.
- Runtime gains depend on layer mix and deployment environment; small models may
  show minimal latency improvements.
