#include "moe_ops.h"

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_softmax", &topk_softmax,
        "Apply topk softmax to the gating outputs.");
  m.def("moe_align_block_size", &moe_align_block_size,
          "Aligning the number of tokens to be processed by each expert such "
          "that it is divisible by the block size.");
  m.def("silu_and_mul", &silu_and_mul, "Activation function used in SwiGLU.");
}
