from test_common import checkAllclose, perftest
import torch
import torch.nn.functional as F
import triton.language as tl
import numpy as np
import sys
import os
from typing import Any, Callable, Dict, Optional, Tuple
import rocmKernels
print(rocmKernels.__file__, 11111111111111)
print(dir(rocmKernels))
if 1:
    _path = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, f'{_path}/../../')
    from rocm_kernels.fused_moe_gelu import fused_topk, moe_align_block_size, fused_experts
    from rocm_kernels.fused_moe_bf16_asm import asm_moe, moe_sorting_ck

BLOCK_SIZE_M = 32


@perftest()
def moe_sorting_vllm(topk_ids: torch.Tensor,
                     block_size: int,
                     num_experts: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk = topk_ids.shape[1]
    # max_num_tokens_padded = (
    #     topk_ids.numel() + num_experts * (block_size - 1)+block_size-1)//block_size*block_size
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    # max_num_tokens_padded = int(
    #     (max_num_tokens_padded+block_size-1)//block_size*block_size)
    max_num_m_blocks = int((max_num_tokens_padded+block_size-1)//block_size)

    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.shape[0]*topk)
    expert_ids = torch.empty((max_num_m_blocks, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    token_nums = torch.empty((max_num_m_blocks, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)
    rocmKernels.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                                     expert_ids, token_nums, num_tokens_post_pad)
    return sorted_ids, expert_ids, token_nums, num_tokens_post_pad


@perftest()
def moe_sorting_ck_test(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype):
    return moe_sorting_ck(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype)


def test_moe_sort(dtype, token, model_dim, hidden_dim, E, topk):
    dim = (token, model_dim, hidden_dim)
    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    w1 = torch.randn((E, hidden_dim, model_dim), dtype=dtype, device="cuda")
    w2 = torch.randn((E, model_dim, hidden_dim), dtype=dtype, device="cuda")
    score = torch.randn((token, E), device="cuda", dtype=dtype)

    topk_weights, topk_ids = fused_topk(input, score, topk, True)
    # print(f'{topk_weights=}')
    # print(f'{topk_ids=}')

    (sorted_ids_a,
     sorted_expert_ids_a,
     token_nums,
     num_tokens_post_padded_a), avg_a = moe_sorting_vllm(
        topk_ids, BLOCK_SIZE_M, E)
    sorted_ids_a = sorted_ids_a//topk

    (sorted_ids_b,
     sorted_weights_b,
     sorted_expert_ids_b,
     num_tokens_post_padded_b,
     moe_buf), avg_b = moe_sorting_ck_test(topk_ids, topk_weights, E,
                                           model_dim, dtype)
    # print(f'{num_tokens_post_padded_a=}')
    # print(f'{num_tokens_post_padded_b=}')
    # print(f'{sorted_ids_a.shape=}')
    # print(f'{sorted_ids_b.shape=}')
    # pad_a = (sorted_ids_a.shape[0]+BLOCK_SIZE_M -
    #          1)//BLOCK_SIZE_M*BLOCK_SIZE_M-sorted_ids_a.shape[0]
    # pad_b = (sorted_ids_b.shape[0]+BLOCK_SIZE_M -
    #          1)//BLOCK_SIZE_M*BLOCK_SIZE_M-sorted_ids_b.shape[0]
    # print(f'{F.pad(sorted_ids_a,(0,pad_a), "constant", 0).view(-1,BLOCK_SIZE_M)=}')
    # print(f'{F.pad(sorted_ids_b,(0,pad_b), "constant", 0).view(-1,BLOCK_SIZE_M)=}')
    # print(f'{sorted_expert_ids_a=}')
    # print(f'{sorted_expert_ids_b=}')
    # print(f'{moe_buf.max()=}')

    print(
        f"[perf] {token=}, {model_dim=}, {hidden_dim=}, {E=}, {topk=}, dtype: {dtype}, torch avg: {avg_a:.2f} us, ck avg: {avg_b:.2f} us, uplift: {avg_a/avg_b-1:.1%}", end=' ')
    if num_tokens_post_padded_a[0] != num_tokens_post_padded_b[0]:
        print("[F!!!]")
        return
    checkAllclose(num_tokens_post_padded_a, num_tokens_post_padded_b, atol=0)
    checkAllclose(sorted_ids_a[:num_tokens_post_padded_a[0]],
                  sorted_ids_b[:num_tokens_post_padded_b[0]])
    checkAllclose(sorted_expert_ids_a[:num_tokens_post_padded_a[0]//BLOCK_SIZE_M],
                  sorted_expert_ids_b[:num_tokens_post_padded_b[0]//BLOCK_SIZE_M])
    print(f"[passed~]")


# print('test test_moe_sort')
# for dtype in [torch.float16, torch.bfloat16][1:]:
#     for m in [1, 2, 4, 8, 16, 32, 64, 128, 256][3:]:
#         for dim in [4096, 8192, 16384, 32768, 65536][:-2]:
#             for hdim in [1024, 4096, 8192, 16384, 32768, 65536][:-2]:
#                 test_moe_sort(dtype, m, dim, hdim, 32, 5)


def permute_weight_a(x: torch.Tensor) -> torch.Tensor:
    # Hardcode BLOCK_K and BLOCK_N
    BK = 128
    BN = 128
    x_ = x
    x_ = x_.view(x.shape[0],
                 x.shape[1]//BN, BN//16, 16,
                 x.shape[2]//BK, BK//32, 4, 8)
    x_ = x_.permute(0, 1, 5, 2, 6, 4, 3, 7)
    x_ = x_.contiguous()
    x_ = x_.view(x.shape[0], x.shape[1], x.shape[2])
    return x_


def permute_weight_b(x: torch.Tensor) -> torch.Tensor:
    # Hardcode BLOCK_K and BLOCK_N
    BK = 32
    BN = 16
    x_ = x
    x_ = x_.view(x.shape[0],
                 x.shape[1]//BN, BN,
                 x.shape[2]//BK, 4, 8)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(x.shape[0], x.shape[1], x.shape[2])
    return x_


@perftest()
def torch_moe(hidden_states, w1, w2, topk_weight, topk_ids):
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    hidden_states = hidden_states.view(
        B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(
        B * topk,
        w2.shape[1],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            silu_input = hidden_states[mask] @ (w1[i].transpose(0, 1))
            d = silu_input.shape[-1] // 2
            silu_output_shape = silu_input.shape[:-1] + (d,)
            silu_out = torch.empty(
                silu_output_shape, dtype=silu_input.dtype, device=silu_input.device
            )
            silu_out = F.gelu(silu_input)
            out[mask] = silu_out @ (w2[i].transpose(0, 1))
    # out = out + 2.0
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


@perftest()
def asm_moe_test(hidden_states, w1, w2, topk_weight, topk_ids):
    return asm_moe(hidden_states, w1, w2, topk_weight, topk_ids)


@perftest()
def vllm_moe(hidden_states, w1, w2, topk_weight, topk_ids):
    return fused_experts(hidden_states,
                         w1,
                         w2,
                         topk_weight,
                         topk_ids,
                         inplace=False)


def test_fmoe(dtype, token, model_dim, hidden_dim, E, topk):
    dim = (token, model_dim, hidden_dim)
    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    w1 = torch.randn((E, hidden_dim, model_dim), dtype=dtype, device="cuda")
    w2 = torch.randn((E, model_dim, hidden_dim), dtype=dtype, device="cuda")
    score = torch.randn((token, E), device="cuda", dtype=dtype)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    # ref implement
    # w1a = permute_weight_a(w1)
    # w2a = permute_weight_a(w2)
    w1a = w1
    w2a = w2
    ref1, avg_a = vllm_moe(input,
                           w1a,
                           w2a,
                           topk_weights,
                           topk_ids)
    # print(f'{ref1=}')
    # ref2 implement
    ref2, avg_c = torch_moe(input,
                            w1,
                            w2,
                            topk_weights,
                            topk_ids)
    # print(f'{ref2=}')

    # b implement
    w1b = permute_weight_b(w1)
    w2b = permute_weight_b(w2)
    out_b, avg_b = asm_moe_test(input, w1b, w2b, topk_weights, topk_ids)

    # print(f'{out_b=}')
    print(
        f"[perf] {token=}, {model_dim=}, {hidden_dim=}, {E=}, {topk=}, dtype: {dtype}, torch_avg: {avg_a:.2f} us, asm_avg: {avg_b:.2f} us,smtorch_k_avg: {avg_c:.2f} us, uplift: {avg_a/avg_b-1:.1%}", end=' ')
    # checkAllclose(ref1, ref2, rtol=0.05, atol=20)
    checkAllclose(ref2, out_b, rtol=0.01, atol=10)


print('test test_fmoe')
for dtype in [torch.float16, torch.bfloat16][1:]:
    for m in [1, 2, 4, 8, 16, 26, 32, 64, 128, 160, 192, 224, 256]:
        for dim in [4096, 8192, 16384, 32768, 65536][1:1+1]:
            for hdim in [1024, 4096, 8192, 16384, 32768, 65536][:1]:
                test_fmoe(dtype, m, dim, hdim, 32, 5)
