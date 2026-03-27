load("@xla//third_party/rules_python/python:defs.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")

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
        "//tensorflow/python/keras:activations",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/initializers",
        "//tensorflow/python/keras/layers:convolutional",
        "//tensorflow/python/keras/layers:embeddings",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/ops:array_ops",
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
        "//tensorflow/python/keras:models",
        "//tensorflow/python/keras/layers:convolutional",
        "//tensorflow/python/keras/layers:core",
        "//tensorflow/python/keras/layers:embeddings",
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
        ":config",
        ":core",
        ":keras",
    ],
)

py_test(
    name = "core_test",
    srcs = ["core_test.py"],
    deps = [
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
