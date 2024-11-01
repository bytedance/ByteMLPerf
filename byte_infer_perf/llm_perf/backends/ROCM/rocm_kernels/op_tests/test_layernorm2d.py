import torch
import torch.nn.functional as F
import numpy as np
import rocmKernels

num_iters = 100
num_warmup = 20


def run_torch(input, weight, bias, eps, residual=None):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies = []
    for i in range(num_iters+num_warmup):
        start_event.record()
        if residual is None:
            residual_out = None
            output = F.layer_norm(
                input=input,
                normalized_shape=(input.shape[-1],),
                weight=weight,
                bias=bias,
                eps=eps
            )
        else:
            residual_out = input + residual
            output = F.layer_norm(
                input=residual_out,
                normalized_shape=(input.shape[-1],),
                weight=weight,
                bias=bias,
                eps=eps
            )
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = np.mean(latencies[num_warmup:]) * 1000  # us
    # print(f"run_torch avg time: {avg} us")
    return output, avg, residual_out


def run_ck(input, weight, bias, eps, residual=None):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies = []
    for i in range(num_iters+num_warmup):
        start_event.record()
        if residual is None:
            residual_out = None
            output = torch.empty_like(input)
            rocmKernels.layernorm2d_fwd(
                output,
                input,
                weight,
                bias,
                eps
            )
        else:
            residual_out = torch.empty_like(input)
            output = torch.empty_like(input)
            rocmKernels.layernorm2d_fwd_with_add(
                output,
                input,
                residual,
                residual_out,
                weight,
                bias,
                eps
            )
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = np.mean(latencies[num_warmup:]) * 1000  # us
    # print(f"run_ck    avg time: {avg} us")
    return output, avg, residual_out


def checkAllclose(a, b, rtol=1e-2, atol=1e-2):
    assert torch.allclose(
        a, b, rtol, atol), f'''torch and ck results are not close\ntorch: {a.shape}\n{a}\nck: {b.shape}\n{b}\nmax delta:{(a-b).max()}
        detail delta: {(a-b)}'''


def test_layernorm2d(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    a, avg_a, *_ = run_torch(input, weight, bias, 1e-5)
    b, avg_b, *_ = run_ck(input, weight, bias, 1e-5)
    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:.2f} us, ck avg: {avg_b:.2f} us, uplift: {avg_a/avg_b-1:.1%}", end=' ')
    checkAllclose(a, b)
    print(f"[passed~]")


def test_layernorm2d_fuseAdd(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    res = torch.randn(dim, dtype=dtype, device="cuda")
    a, avg_a, res_a, *_ = run_torch(input, weight, bias, 1e-5, residual=res)
    b, avg_b, res_b, *_ = run_ck(input, weight, bias, 1e-5, residual=res)

    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:.2f} us, ck avg: {avg_b:.2f} us, uplift: {avg_a/avg_b-1:.1%}", end=' ')
    checkAllclose(a, b, rtol=1e-2, atol=1e-1)
    checkAllclose(res_a, res_b)
    print(f" [passed~]")


for dtype in [torch.float16, torch.bfloat16]:
    for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        for n in [4096, 8192, 16384, 32768, 65536]:
            test_layernorm2d(dtype, m, n)


print('\nstart fuse add test')
for dtype in [torch.float16, torch.bfloat16]:
    for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        for n in [4096, 8192, 16384, 32768, 65536]:
            test_layernorm2d_fuseAdd(dtype, m, n)
