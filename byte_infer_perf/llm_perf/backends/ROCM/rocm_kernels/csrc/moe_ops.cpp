#include "moe_ops.h"

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("topk_softmax", &topk_softmax,
        "Apply topk softmax to the gating outputs.");
  m.def("moe_align_block_size", &moe_align_block_size,
        "Aligning the number of tokens to be processed by each expert such "
        "that it is divisible by the block size.");
  m.def("silu_and_mul", &silu_and_mul, "Activation function used in SwiGLU.");
  m.def("rms_norm", &rms_norm, "Apply Root Mean Square (RMS) Normalization to the input tensor.");
  m.def("fused_add_rms_norm", &fused_add_rms_norm, "In-place fused Add and RMS Normalization");
  m.def("wvSpltK", &wvSpltK, "wvSpltK(Tensor in_a, Tensor in_b, Tensor! out_c, int N_in,"
                             "        int CuCount) -> ()");
}
