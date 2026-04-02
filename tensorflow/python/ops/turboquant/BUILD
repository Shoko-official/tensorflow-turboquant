load("@xla//third_party/rules_python/python:defs.bzl", "py_binary", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test", "tf_custom_op_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    srcs_version = "PY3",
    strict_deps = False,
)

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    strict_deps = False,
    deps = [
        ":config",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "calibration",
    srcs = ["calibration.py"],
    srcs_version = "PY3",
    strict_deps = False,
    deps = [
        ":config",
        "//third_party/py/numpy",
        "//tensorflow/python/keras:models",
    ],
)

py_library(
    name = "keras",
    srcs = ["keras.py"],
    srcs_version = "PY3",
    strict_deps = False,
    deps = [
        ":config",
        ":core",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:activations",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:embedding_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:standard_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "api",
    srcs = ["api.py"],
    srcs_version = "PY3",
    strict_deps = False,
    deps = [
        ":calibration",
        ":config",
        ":core",
        ":keras",
        "//third_party/py/numpy",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:models",
        "//tensorflow/python/saved_model",
    ],
)

py_library(
    name = "turboquant",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    strict_deps = False,
    deps = [
        ":api",
        ":calibration",
        ":cpp_ops",
        ":config",
        ":core",
        ":keras",
    ],
)

py_library(
    name = "benchmark_lib",
    srcs = ["benchmark_turboquant.py"],
    srcs_version = "PY3",
    strict_deps = False,
    deps = [
        ":api",
        ":config",
        "//third_party/py/numpy",
        "//tensorflow/python/keras",
    ],
)

py_library(
    name = "benchmark_real_models_lib",
    srcs = ["benchmark_real_models.py"],
    srcs_version = "PY3",
    strict_deps = False,
    deps = [
        ":api",
        ":config",
        "//third_party/py/numpy",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:models",
    ],
)

py_library(
    name = "analyze_lib",
    srcs = ["analyze_turboquant_results.py"],
    srcs_version = "PY3",
    strict_deps = False,
    deps = [
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "ablation_lib",
    srcs = ["run_turboquant_ablations.py"],
    srcs_version = "PY3",
    strict_deps = False,
    deps = [
        ":benchmark_lib",
        ":config",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "benchmark_turboquant",
    srcs = ["benchmark_turboquant.py"],
    srcs_version = "PY3",
    deps = [":benchmark_lib"],
)

py_binary(
    name = "benchmark_real_models",
    srcs = ["benchmark_real_models.py"],
    srcs_version = "PY3",
    deps = [":benchmark_real_models_lib"],
)

py_library(
    name = "profile_lib",
    srcs = ["profile_turboquant.py"],
    srcs_version = "PY3",
    strict_deps = False,
    deps = [
        ":api",
        ":config",
        "//third_party/py/numpy",
        "//tensorflow/python/keras",
    ],
)

py_binary(
    name = "profile_turboquant",
    srcs = ["profile_turboquant.py"],
    srcs_version = "PY3",
    deps = [":profile_lib"],
)

py_binary(
    name = "analyze_turboquant_results",
    srcs = ["analyze_turboquant_results.py"],
    srcs_version = "PY3",
    deps = [":analyze_lib"],
)

py_binary(
    name = "run_turboquant_ablations",
    srcs = ["run_turboquant_ablations.py"],
    srcs_version = "PY3",
    deps = [":ablation_lib"],
)

tf_custom_op_library(
    name = "turboquant_packed_indices_op.so",
    srcs = ["packed_indices_op_kernel.cc"],
)

py_library(
    name = "cpp_ops",
    srcs = ["cpp_ops.py"],
    data = [":turboquant_packed_indices_op.so"],
    srcs_version = "PY3",
    strict_deps = False,
    deps = ["//tensorflow:tensorflow_py"],
)

py_test(
    name = "core_test",
    srcs = ["core_test.py"],
    deps = [
        ":cpp_ops",
        ":config",
        ":core",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "keras_test",
    srcs = ["keras_test.py"],
    deps = [
        ":api",
        ":config",
        ":core",
        ":keras",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/keras",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "calibration_test",
    srcs = ["calibration_test.py"],
    deps = [
        ":api",
        ":calibration",
        ":config",
        ":keras",
        "//tensorflow/python/keras",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "benchmark_test",
    srcs = ["benchmark_test.py"],
    deps = [
        ":ablation_lib",
        ":analyze_lib",
        ":benchmark_real_models_lib",
        ":benchmark_lib",
        ":config",
        ":profile_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)
