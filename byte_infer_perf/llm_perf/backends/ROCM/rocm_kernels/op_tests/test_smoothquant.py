import torch
import torch.nn.functional as F
import numpy as np
import rocmKernels
from test_common import checkAllclose, perftest
import argparse

num_iters = 100
num_warmup = 20

def pertoken_quant(hidden_states_input, y_scale_dtype, x_scale = None):
    # assume output int8, hidden_states is [m, n] shape and in fp16/bf16
    if x_scale is None:
        hidden_states = hidden_states_input
    else:
        # smooth quant
        hidden_states = hidden_states_input.to(x_scale) * x_scale
    # [m, 1]
    per_token_amax, _ = torch.max(
        input=torch.abs(hidden_states), 
        dim=-1, 
        keepdim=True
    )
    per_token_scale = per_token_amax.to(dtype=torch.float32) / 127.0

    # quant hidden_states
    hidden_states = (hidden_states / per_token_scale).to(dtype=torch.int8)

    return hidden_states, per_token_scale.to(y_scale_dtype)
    # hidden_states now is int8 will feed to next layer as intput
    # per_token_scale will be used as dequant factor later layer


def run_torch(input, x_scale, y_scale_dtype = torch.float32):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies = []
    for i in range(num_iters+num_warmup):
        start_event.record()
        output, y_scale = pertoken_quant(input, x_scale=x_scale, y_scale_dtype=y_scale_dtype)
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = np.mean(latencies[num_warmup:]) * 1000  # us
    # print(f"run_torch avg time: {avg} us")
    return output, y_scale, avg


def run_ck(input, x_scale, y_scale_dtype = torch.float32):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies = []
    for i in range(num_iters+num_warmup):
        start_event.record()
        output = torch.empty(input.shape, device="cuda", dtype=torch.int8)
        y_scale = torch.empty(input.shape[0], 1, device="cuda", dtype=y_scale_dtype)
        rocmKernels.smoothquant_fwd(output,
                                    input,
                                    x_scale,
                                    y_scale)

        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = np.mean(latencies[num_warmup:]) * 1000  # us
    # print(f"run_ck    avg time: {avg} us")
    return output, y_scale, avg


    
def test_Smoothquant_instance(dtype, m, n, xscaleType):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    xscale = torch.randn(n, dtype=xscaleType, device="cuda")
    a, yscale_a, avg_a = run_torch(input, x_scale=xscale)
    b, yscale_b, avg_b = run_ck(input, x_scale=xscale)

    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:.2f} us, ck avg: {avg_b:.2f} us, uplift: {avg_a/avg_b-1:.1%}", end=' ')
    checkAllclose(a, b, rtol=0, atol=1)
    checkAllclose(yscale_a, yscale_b, rtol=1e-3, atol=1e-3)

def test_Smoothquant():
    print('\nstart layernorm2d fuse Smoothquant test')
    for scaleType in [ torch.float32]:
        for dtype in [torch.float16, torch.bfloat16]:
            for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                for n in [10, 4096, 8192]:
                    test_Smoothquant_instance(dtype, m, n, xscaleType=scaleType)

if __name__ == "__main__":
    test_Smoothquant()