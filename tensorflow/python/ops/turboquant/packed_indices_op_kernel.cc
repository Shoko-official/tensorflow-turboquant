/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("TurboQuantUnpackIndices")
    .Input("packed: uint8")
    .Input("flat_size: int32")
    .Input("num_bits: int32")
    .Output("indices: uint8")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle packed;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &packed));
      shape_inference::ShapeHandle flat_size;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &flat_size));
      shape_inference::ShapeHandle num_bits;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &num_bits));

      const Tensor* flat_size_tensor = c->input_tensor(1);
      if (flat_size_tensor != nullptr) {
        const int32 value = flat_size_tensor->scalar<int32>()();
        if (value < 0) {
          return errors::InvalidArgument("`flat_size` must be >= 0. Got: ",
                                         value);
        }
        c->set_output(0, c->Vector(value));
      } else {
        c->set_output(0, c->Vector(c->UnknownDim()));
      }
      return absl::OkStatus();
    })
    .Doc(R"doc(
Unpacks TurboQuant indices from a little-endian bit-packed uint8 payload.

packed: 1-D uint8 packed bitstream.
flat_size: Number of output index elements to unpack.
num_bits: Number of bits per index in [1, 8].
indices: 1-D uint8 unpacked index vector.
)doc");

class TurboQuantUnpackIndicesOp : public OpKernel {
 public:
  explicit TurboQuantUnpackIndicesOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& packed_tensor = context->input(0);
    const Tensor& flat_size_tensor = context->input(1);
    const Tensor& num_bits_tensor = context->input(2);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(packed_tensor.shape()),
                errors::InvalidArgument("`packed` must be rank-1."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(flat_size_tensor.shape()),
                errors::InvalidArgument("`flat_size` must be scalar."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_bits_tensor.shape()),
                errors::InvalidArgument("`num_bits` must be scalar."));

    const int32 flat_size = flat_size_tensor.scalar<int32>()();
    const int32 num_bits = num_bits_tensor.scalar<int32>()();
    OP_REQUIRES(context, flat_size >= 0,
                errors::InvalidArgument("`flat_size` must be >= 0. Got: ",
                                        flat_size));
    OP_REQUIRES(context, num_bits >= 1 && num_bits <= 8,
                errors::InvalidArgument("`num_bits` must be in [1, 8]. Got: ",
                                        num_bits));

    const int64 total_bits = static_cast<int64>(flat_size) * num_bits;
    const int64 needed_bytes = (total_bits + 7) / 8;
    auto packed = packed_tensor.flat<uint8>();
    OP_REQUIRES(
        context, packed.size() >= needed_bytes,
        errors::InvalidArgument("Packed payload is too small. Need ",
                                needed_bytes, " bytes, got ", packed.size(),
                                "."));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({flat_size}),
                                                     &output));
    auto out = output->flat<uint8>();
    if (flat_size == 0) {
      return;
    }

    const int32 max_value = (1 << num_bits) - 1;
    int64 bit_offset = 0;
    for (int64 i = 0; i < static_cast<int64>(flat_size); ++i) {
      const int64 byte_index = bit_offset >> 3;
      const int32 bit_index = static_cast<int32>(bit_offset & 7);
      int32 value = static_cast<int32>(packed(byte_index)) >> bit_index;
      const int32 overflow_bits = bit_index + num_bits - 8;
      if (overflow_bits > 0) {
        value |= static_cast<int32>(packed(byte_index + 1)) << (8 - bit_index);
      }
      out(i) = static_cast<uint8>(value & max_value);
      bit_offset += num_bits;
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("TurboQuantUnpackIndices").Device(DEVICE_CPU),
    TurboQuantUnpackIndicesOp);

}  // namespace tensorflow
