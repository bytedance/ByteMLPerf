import torch
from rocmKernels import gemm_a8w8

def gemm_a8w8_(
        XQ: torch.tensor,
        WQ: torch.tensor,
        x_scale: torch.tensor,
        w_scale: torch.tensor,
        bias=None,
        dtype=torch.bfloat16):
    assert dtype in [torch.bfloat16, torch.float16], \
        f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    Y = torch.empty(m, n, dtype=dtype, device="cuda")
    gemm_a8w8(XQ, WQ, x_scale, w_scale, Y, bias)
    return Y