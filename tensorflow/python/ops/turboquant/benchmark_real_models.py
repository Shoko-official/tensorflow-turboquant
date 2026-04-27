"""Benchmark TurboQuant on realistic model families and dataset inputs."""

import argparse
import json
import math
import time

import numpy as np

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import DepthwiseConv2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import SeparableConv2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import ReLU
from tensorflow.python.keras.models import Model
from tensorflow.python.ops.turboquant.api import quantize_model
from tensorflow.python.ops.turboquant.config import TurboQuantConfig


def _parse_csv_ints(value):
  return tuple(int(item.strip()) for item in value.split(',') if item.strip())


def _residual_block(inputs, filters, stride=1):
  residual = inputs
  x = Conv2D(filters, 3, padding='same', strides=stride, use_bias=False)(inputs)
  x = ReLU()(x)
  x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
  if stride != 1 or int(inputs.shape[-1]) != filters:
    residual = Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(
        residual
    )
  x = Add()([x, residual])
  return ReLU()(x)


def _build_residual_cnn(input_shape, num_classes):
  inputs = Input(shape=input_shape)
  x = Conv2D(32, 3, padding='same', use_bias=False)(inputs)
  x = ReLU()(x)
  x = _residual_block(x, 32, stride=1)
  x = _residual_block(x, 64, stride=2)
  x = _residual_block(x, 128, stride=2)
  x = GlobalAveragePooling2D()(x)
  outputs = Dense(num_classes)(x)
  return Model(inputs, outputs, name='residual_cnn')


def _build_separable_cnn(input_shape, num_classes):
  return Sequential([
      Input(shape=input_shape),
      DepthwiseConv2D(3, padding='same'),
      Conv2D(32, 1, padding='same', activation='relu'),
      SeparableConv2D(64, 3, padding='same', activation='relu'),
      MaxPool2D(pool_size=2),
      SeparableConv2D(128, 3, padding='same', activation='relu'),
      GlobalAveragePooling2D(),
      Dense(num_classes),
  ])


_MODEL_BUILDERS = {
    'residual_cnn': _build_residual_cnn,
    'separable_cnn': _build_separable_cnn,
}


def _load_dataset(
    dataset_source,
    seed,
    sample_count,
    input_shape,
    num_classes,
    dataset_npz_path='',
):
  if dataset_source == 'synthetic':
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(sample_count,) + tuple(input_shape)).astype(
        np.float32
    )
    labels = rng.integers(0, num_classes, size=(sample_count,), dtype=np.int32)
    return features, labels

  if dataset_source == 'npz':
    if not dataset_npz_path:
      raise ValueError(
          '`dataset_npz_path` is required when dataset_source="npz".'
      )
    loaded = np.load(dataset_npz_path)
    if 'x' not in loaded:
      raise ValueError('NPZ dataset must contain an `x` array.')
    features = np.asarray(loaded['x'], dtype=np.float32)
    labels = np.asarray(
        loaded['y'], dtype=np.int32
    ) if 'y' in loaded else np.zeros((features.shape[0],), dtype=np.int32)
    if features.shape[0] != labels.shape[0]:
      raise ValueError('NPZ dataset arrays `x` and `y` must share first dimension.')
    if features.shape[1:] != tuple(input_shape):
      raise ValueError(
          'NPZ dataset feature shape mismatch. '
          f'Expected {(None,) + tuple(input_shape)}, got {features.shape}.'
      )
    if sample_count and features.shape[0] > sample_count:
      features = features[:sample_count]
      labels = labels[:sample_count]
    return features, labels

  raise ValueError(
      f'Unknown dataset_source={dataset_source!r}. Supported: synthetic, npz.'
  )


def _measure_latency_ms(model, inputs, warmup_steps, benchmark_steps):
  for _ in range(warmup_steps):
    model(inputs, training=False)
  samples = []
  for _ in range(benchmark_steps):
    start = time.perf_counter()
    model(inputs, training=False)
    samples.append((time.perf_counter() - start) * 1e3)
  latencies = np.asarray(samples, dtype=np.float64)
  return {
      'mean': float(np.mean(latencies)),
      'p50': float(np.percentile(latencies, 50)),
      'p95': float(np.percentile(latencies, 95)),
  }


def _evaluate_model(logits, labels):
  predicted = np.argmax(logits, axis=-1)
  return {
      'accuracy': float(np.mean(predicted == labels)),
      'logits_mean_abs': float(np.mean(np.abs(logits))),
      'logits_std': float(np.std(logits)),
  }


def _summarize_metric(values):
  values = np.asarray(values, dtype=np.float64)
  if values.size == 0:
    raise ValueError('Expected at least one value to summarize.')
  ci95_half_width = 0.0
  if values.size > 1:
    ci95_half_width = 1.96 * float(np.std(values, ddof=1)) / math.sqrt(values.size)
  return {
      'count': int(values.size),
      'mean': float(np.mean(values)),
      'min': float(np.min(values)),
      'max': float(np.max(values)),
      'stddev': float(np.std(values, ddof=0)),
      'confidence_interval_95': {
          'lower': float(np.mean(values) - ci95_half_width),
          'upper': float(np.mean(values) + ci95_half_width),
          'half_width': float(ci95_half_width),
      },
  }


def _speedup_summary(results, key):
  speedups = []
  for item in results:
    float_p50 = item['latency_ms']['float']['p50']
    quantized_p50 = item['latency_ms'][key]['p50']
    speedups.append(float_p50 / max(quantized_p50, 1e-9))
  return _summarize_metric(speedups)


def _build_summary(results):
  turbo_ratios = [
      float(item['compression']['turboquant']['effective_compression_ratio'])
      for item in results
  ]
  baseline_ratios = [
      float(item['compression']['baseline']['effective_compression_ratio'])
      for item in results
  ]
  turbo_agreement = [item['drift']['argmax_agreement_turboquant'] for item in results]
  baseline_agreement = [item['drift']['argmax_agreement_baseline'] for item in results]
  turbo_accuracy_delta = [
      item['metric_deltas']['turboquant_accuracy_delta'] for item in results
  ]
  baseline_accuracy_delta = [
      item['metric_deltas']['baseline_accuracy_delta'] for item in results
  ]
  turbo_mse = [item['drift']['turboquant_mse'] for item in results]
  baseline_mse = [item['drift']['baseline_mse'] for item in results]

  return {
      'case_count': int(len(results)),
      'compression_ratio': {
          'turboquant': _summarize_metric(turbo_ratios),
          'baseline': _summarize_metric(baseline_ratios),
      },
      'argmax_agreement': {
          'turboquant': _summarize_metric(turbo_agreement),
          'baseline': _summarize_metric(baseline_agreement),
      },
      'accuracy_delta': {
          'turboquant': _summarize_metric(turbo_accuracy_delta),
          'baseline': _summarize_metric(baseline_accuracy_delta),
      },
      'output_mse': {
          'turboquant': _summarize_metric(turbo_mse),
          'baseline': _summarize_metric(baseline_mse),
      },
      'latency_speedup_p50_float_over_quantized': {
          'turboquant': _speedup_summary(results, 'turboquant'),
          'baseline': _speedup_summary(results, 'baseline'),
      },
  }


def validate_quality_gates(
    report,
    *,
    max_turboquant_accuracy_drop=None,
    max_baseline_accuracy_drop=None,
    min_turboquant_argmax_agreement=None,
    min_baseline_argmax_agreement=None,
    max_turboquant_mse=None,
):
  """Returns a list of task-level quality gate failures for a benchmark report."""
  failures = []
  summary = report['summary']
  turbo_accuracy_min = summary['accuracy_delta']['turboquant']['min']
  baseline_accuracy_min = summary['accuracy_delta']['baseline']['min']
  turbo_agreement_min = summary['argmax_agreement']['turboquant']['min']
  baseline_agreement_min = summary['argmax_agreement']['baseline']['min']
  turbo_mse_max = summary['output_mse']['turboquant']['max']

  if (
      max_turboquant_accuracy_drop is not None
      and turbo_accuracy_min < -float(max_turboquant_accuracy_drop)
  ):
    failures.append(
        'TurboQuant accuracy drop gate failed: '
        f'min delta={turbo_accuracy_min:.4f} < '
        f'-{float(max_turboquant_accuracy_drop):.4f}.'
    )
  if (
      max_baseline_accuracy_drop is not None
      and baseline_accuracy_min < -float(max_baseline_accuracy_drop)
  ):
    failures.append(
        'Baseline accuracy drop gate failed: '
        f'min delta={baseline_accuracy_min:.4f} < '
        f'-{float(max_baseline_accuracy_drop):.4f}.'
    )
  if (
      min_turboquant_argmax_agreement is not None
      and turbo_agreement_min < float(min_turboquant_argmax_agreement)
  ):
    failures.append(
        'TurboQuant argmax agreement gate failed: '
        f'min agreement={turbo_agreement_min:.4f} < '
        f'{float(min_turboquant_argmax_agreement):.4f}.'
    )
  if (
      min_baseline_argmax_agreement is not None
      and baseline_agreement_min < float(min_baseline_argmax_agreement)
  ):
    failures.append(
        'Baseline argmax agreement gate failed: '
        f'min agreement={baseline_agreement_min:.4f} < '
        f'{float(min_baseline_argmax_agreement):.4f}.'
    )
  if max_turboquant_mse is not None and turbo_mse_max > float(max_turboquant_mse):
    failures.append(
        'TurboQuant output MSE gate failed: '
        f'max mse={turbo_mse_max:.6f} > {float(max_turboquant_mse):.6f}.'
    )
  return failures


def _single_run(
    model_name,
    config,
    baseline_config,
    dataset_source,
    dataset_npz_path,
    seed,
    train_epochs,
    sample_count,
    eval_count,
    batch_size,
    warmup_steps,
    benchmark_steps,
    num_classes,
):
  input_shape = (32, 32, 3)
  builder = _MODEL_BUILDERS[model_name]
  model = builder(input_shape, num_classes)
  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )
  features, labels = _load_dataset(
      dataset_source=dataset_source,
      seed=seed,
      sample_count=max(sample_count, eval_count),
      input_shape=input_shape,
      num_classes=num_classes,
      dataset_npz_path=dataset_npz_path,
  )
  if sample_count and features.shape[0] > sample_count:
    features = features[:sample_count]
    labels = labels[:sample_count]
  if train_epochs > 0:
    model.fit(
        features,
        labels,
        batch_size=batch_size,
        epochs=train_epochs,
        verbose=0,
    )

  eval_features = features[:eval_count]
  eval_labels = labels[:eval_count]
  model(eval_features[:1], training=False)

  turbo_report = quantize_model(model, config, return_report=True)
  turbo_model = turbo_report['model']
  baseline_report = quantize_model(model, baseline_config, return_report=True)
  baseline_model = baseline_report['model']

  float_logits = model(eval_features, training=False).numpy()
  turbo_logits = turbo_model(eval_features, training=False).numpy()
  baseline_logits = baseline_model(eval_features, training=False).numpy()

  float_metrics = _evaluate_model(float_logits, eval_labels)
  turbo_metrics = _evaluate_model(turbo_logits, eval_labels)
  baseline_metrics = _evaluate_model(baseline_logits, eval_labels)

  float_latency = _measure_latency_ms(
      model, eval_features, warmup_steps=warmup_steps, benchmark_steps=benchmark_steps
  )
  turbo_latency = _measure_latency_ms(
      turbo_model,
      eval_features,
      warmup_steps=warmup_steps,
      benchmark_steps=benchmark_steps,
  )
  baseline_latency = _measure_latency_ms(
      baseline_model,
      eval_features,
      warmup_steps=warmup_steps,
      benchmark_steps=benchmark_steps,
  )

  turbo_drift = float_logits - turbo_logits
  baseline_drift = float_logits - baseline_logits
  return {
      'model_name': model_name,
      'dataset_source': dataset_source,
      'seed': int(seed),
      'train_epochs': int(train_epochs),
      'sample_count': int(features.shape[0]),
      'eval_count': int(eval_features.shape[0]),
      'quantization_config': config.to_dict(),
      'baseline_quantization_config': baseline_config.to_dict(),
      'float_metrics': float_metrics,
      'turboquant_metrics': turbo_metrics,
      'baseline_metrics': baseline_metrics,
      'metric_deltas': {
          'turboquant_accuracy_delta': (
              turbo_metrics['accuracy'] - float_metrics['accuracy']
          ),
          'baseline_accuracy_delta': (
              baseline_metrics['accuracy'] - float_metrics['accuracy']
          ),
      },
      'drift': {
          'turboquant_mse': float(np.mean(np.square(turbo_drift))),
          'turboquant_max_abs': float(np.max(np.abs(turbo_drift))),
          'baseline_mse': float(np.mean(np.square(baseline_drift))),
          'baseline_max_abs': float(np.max(np.abs(baseline_drift))),
          'argmax_agreement_turboquant': float(
              np.mean(
                  np.argmax(float_logits, axis=-1) == np.argmax(turbo_logits, axis=-1)
              )
          ),
          'argmax_agreement_baseline': float(
              np.mean(
                  np.argmax(float_logits, axis=-1)
                  == np.argmax(baseline_logits, axis=-1)
              )
          ),
      },
      'compression': {
          'turboquant': turbo_report['aggregate'],
          'baseline': baseline_report['aggregate'],
      },
      'latency_ms': {
          'float': float_latency,
          'turboquant': turbo_latency,
          'baseline': baseline_latency,
      },
  }


def run_real_model_benchmark(
    models,
    seeds,
    config,
    baseline_config,
    dataset_source='synthetic',
    dataset_npz_path='',
    train_epochs=0,
    sample_count=1024,
    eval_count=256,
    batch_size=32,
    warmup_steps=5,
    benchmark_steps=20,
    num_classes=10,
):
  """Runs a benchmark matrix over realistic model families."""
  results = []
  for model_name in models:
    if model_name not in _MODEL_BUILDERS:
      raise ValueError(
          f'Unknown model_name={model_name!r}. '
          f'Available: {sorted(_MODEL_BUILDERS.keys())}.'
      )
    for seed in seeds:
      results.append(
          _single_run(
              model_name=model_name,
              config=config,
              baseline_config=baseline_config,
              dataset_source=dataset_source,
              dataset_npz_path=dataset_npz_path,
              seed=seed,
              train_epochs=train_epochs,
              sample_count=sample_count,
              eval_count=eval_count,
              batch_size=batch_size,
              warmup_steps=warmup_steps,
              benchmark_steps=benchmark_steps,
              num_classes=num_classes,
          )
      )

  return {
      'metadata': {
          'models': list(models),
          'seeds': [int(seed) for seed in seeds],
          'dataset_source': dataset_source,
          'dataset_npz_path': dataset_npz_path,
          'train_epochs': int(train_epochs),
          'sample_count': int(sample_count),
          'eval_count': int(eval_count),
          'batch_size': int(batch_size),
          'warmup_steps': int(warmup_steps),
          'benchmark_steps': int(benchmark_steps),
          'num_classes': int(num_classes),
      },
      'summary': _build_summary(results),
      'results': results,
  }


def _print_report(report):
  print('TurboQuant real-model benchmark')
  print(
      f"Cases: {report['summary']['case_count']} | "
      f"Models: {', '.join(report['metadata']['models'])} | "
      f"Dataset: {report['metadata']['dataset_source']}"
  )
  print(
      'Compression ratio mean: '
      f"turboquant={report['summary']['compression_ratio']['turboquant']['mean']:.2f}x, "
      f"baseline={report['summary']['compression_ratio']['baseline']['mean']:.2f}x"
  )
  print(
      'Argmax agreement mean: '
      f"turboquant={report['summary']['argmax_agreement']['turboquant']['mean']:.4f}, "
      f"baseline={report['summary']['argmax_agreement']['baseline']['mean']:.4f}"
  )
  print(
      'Accuracy delta mean: '
      f"turboquant={report['summary']['accuracy_delta']['turboquant']['mean']:.4f}, "
      f"baseline={report['summary']['accuracy_delta']['baseline']['mean']:.4f}"
  )


def main():
  parser = argparse.ArgumentParser(
      description='Run TurboQuant benchmark on realistic model families.'
  )
  parser.add_argument(
      '--models',
      type=str,
      default='residual_cnn,separable_cnn',
      help='Comma-separated model names.',
  )
  parser.add_argument('--seeds', type=str, default='123,456,789')
  parser.add_argument(
      '--dataset_source',
      type=str,
      default='synthetic',
      choices=['synthetic', 'npz'],
  )
  parser.add_argument('--dataset_npz_path', type=str, default='')
  parser.add_argument('--num_bits', type=int, default=4)
  parser.add_argument('--group_size', type=int, default=8)
  parser.add_argument('--outlier_threshold', type=float, default=6.0)
  parser.add_argument('--baseline_num_bits', type=int, default=4)
  parser.add_argument('--baseline_group_size', type=int, default=64)
  parser.add_argument('--baseline_outlier_threshold', type=float, default=0.0)
  parser.add_argument('--train_epochs', type=int, default=0)
  parser.add_argument('--sample_count', type=int, default=1024)
  parser.add_argument('--eval_count', type=int, default=256)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--warmup_steps', type=int, default=5)
  parser.add_argument('--benchmark_steps', type=int, default=20)
  parser.add_argument('--num_classes', type=int, default=10)
  parser.add_argument('--max_turboquant_accuracy_drop', type=float, default=None)
  parser.add_argument('--max_baseline_accuracy_drop', type=float, default=None)
  parser.add_argument('--min_turboquant_argmax_agreement', type=float, default=None)
  parser.add_argument('--min_baseline_argmax_agreement', type=float, default=None)
  parser.add_argument('--max_turboquant_mse', type=float, default=None)
  parser.add_argument('--json_output', type=str, default='')
  args = parser.parse_args()

  report = run_real_model_benchmark(
      models=tuple(item.strip() for item in args.models.split(',') if item.strip()),
      seeds=_parse_csv_ints(args.seeds),
      config=TurboQuantConfig(
          num_bits=args.num_bits,
          group_size=args.group_size,
          outlier_threshold=args.outlier_threshold,
      ),
      baseline_config=TurboQuantConfig(
          num_bits=args.baseline_num_bits,
          group_size=args.baseline_group_size,
          outlier_threshold=args.baseline_outlier_threshold,
      ),
      dataset_source=args.dataset_source,
      dataset_npz_path=args.dataset_npz_path,
      train_epochs=args.train_epochs,
      sample_count=args.sample_count,
      eval_count=args.eval_count,
      batch_size=args.batch_size,
      warmup_steps=args.warmup_steps,
      benchmark_steps=args.benchmark_steps,
      num_classes=args.num_classes,
  )
  _print_report(report)
  failures = validate_quality_gates(
      report,
      max_turboquant_accuracy_drop=args.max_turboquant_accuracy_drop,
      max_baseline_accuracy_drop=args.max_baseline_accuracy_drop,
      min_turboquant_argmax_agreement=args.min_turboquant_argmax_agreement,
      min_baseline_argmax_agreement=args.min_baseline_argmax_agreement,
      max_turboquant_mse=args.max_turboquant_mse,
  )
  if failures:
    for failure in failures:
      print(f'QUALITY_GATE_FAILURE: {failure}')
    raise SystemExit(1)
  if args.json_output:
    with open(args.json_output, 'w', encoding='utf-8') as output_file:
      json.dump(report, output_file, indent=2, sort_keys=True)


if __name__ == '__main__':
  main()
