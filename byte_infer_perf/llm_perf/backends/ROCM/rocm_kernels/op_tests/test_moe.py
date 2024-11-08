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
if 1:
    _path = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, f'{_path}/../../')
    from rocm_kernels.fused_moe import fused_topk, moe_align_block_size, fused_experts

BLOCK_SIZE_M = 32


@perftest()
def run_torch(hidden_states, w1, w2, topk_weights, topk_ids, sorted_token_ids, sorted_expert_ids, token_nums, num_tokens_post_padded,
              inplace=False,
              use_fp8_w8a8: bool = False,
              use_int8_w8a16: bool = False,
              w1_scale: Optional[torch.Tensor] = None,
              w2_scale: Optional[torch.Tensor] = None,
              a1_scale: Optional[torch.Tensor] = None,
              a2_scale: Optional[torch.Tensor] = None):
    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    M = num_tokens
    compute_type = tl.bfloat16

    intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache2 = torch.empty((M * topk_ids.shape[1], N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache3 = torch.empty((M, topk_ids.shape[1], w2.shape[1]),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    invoke_fused_moe_kernel(hidden_states,
                            w1,
                            intermediate_cache1,
                            a1_scale,
                            w1_scale,
                            topk_weights,
                            topk_ids,
                            sorted_token_ids,
                            sorted_expert_ids,
                            token_nums,
                            num_tokens_post_padded,
                            False,
                            topk_ids.shape[1],
                            config,
                            compute_type=compute_type,
                            use_fp8_w8a8=use_fp8_w8a8,
                            use_int8_w8a16=use_int8_w8a16)

    moe_kernels.silu_and_mul(
        intermediate_cache2, intermediate_cache1.view(-1, N))

    invoke_fused_moe_kernel(intermediate_cache2,
                            w2,
                            intermediate_cache3,
                            a2_scale,
                            w2_scale,
                            topk_weights,
                            topk_ids,
                            sorted_token_ids,
                            sorted_expert_ids,
                            token_nums,
                            num_tokens_post_padded,
                            True,
                            1,
                            config,
                            compute_type=compute_type,
                            use_fp8_w8a8=use_fp8_w8a8,
                            use_int8_w8a16=use_int8_w8a16)

    moe_kernels.moe_sum(intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx])


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
def run_B(input, w1, w2, sorted_token_ids, sorted_weight_buf, sorted_expert_ids):
    output = torch.empty_like(input)
    rocmKernels.fmoe(
        output,
        input,
        w1,
        w2,
        sorted_token_ids,
        sorted_weight_buf,
        sorted_expert_ids,
    )
    return output


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


def moe_sorting_ck(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype):
    block_size = BLOCK_SIZE_M
    device = topk_ids.device
    topk = topk_ids.shape[1]
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    # max_num_tokens_padded = int(
    #     (max_num_tokens_padded+block_size-1)//block_size*block_size)
    max_num_m_blocks = int(
        (max_num_tokens_padded+block_size-1)//block_size)

    # max_num_tokens_padded = (
    #     topk_ids.numel()+BLOCK_SIZE_M-1) // BLOCK_SIZE_M*BLOCK_SIZE_M * num_experts
    # max_num_tokens_padded = (topk_ids.numel(
    # ) + num_experts * (BLOCK_SIZE_M - 1)+BLOCK_SIZE_M-1)//BLOCK_SIZE_M*BLOCK_SIZE_M
    # print(max_num_tokens_padded)

    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=device)
    sorted_weights = torch.empty((max_num_tokens_padded, ),
                                 dtype=torch.int32,
                                 device=device)
    # sorted_ids.fill_(topk_ids.numel())

    sorted_expert_ids = torch.empty((max_num_m_blocks, ),
                                    dtype=torch.int32,
                                    device=device)

    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=device)
    moe_buf = torch.empty((topk_ids.shape[0], model_dim),
                          dtype=moebuf_dtype,
                          device=device)
    rocmKernels.moe_sorting(topk_ids, topk_weights, sorted_ids, sorted_weights,  sorted_expert_ids,
                            num_tokens_post_pad, moe_buf, num_experts, BLOCK_SIZE_M)
    return sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad, moe_buf


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


def test_fmoe(dtype, token, model_dim, hidden_dim, E, topk):
    dim = (token, model_dim, hidden_dim)
    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    w1 = torch.randn((E, hidden_dim, model_dim), dtype=dtype, device="cuda")
    w2 = torch.randn((E, model_dim, hidden_dim), dtype=dtype, device="cuda")
    score = torch.randn((token, E), device="cuda", dtype=dtype)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    # ref implement
    w1a = permute_weight_a(w1)
    w2a = permute_weight_a(w2)
    out_a = fused_experts(input,
                          w1a,
                          w2a,
                          topk_weights,
                          topk_ids,
                          inplace=False)
    print(f'{out_a=}')

    # b implement
    w1b = permute_weight_b(w1)
    w2b = permute_weight_b(w2)
    sorted_ids_b, sorted_weights_b, sorted_expert_ids_b, num_tokens_post_padded, moe_buf = moe_sorting_ck(topk_ids, topk_weights, E,
                                                                                                          model_dim, dtype)
    rocmKernels.fmoe(moe_buf, input, w1b, w2b, sorted_ids_b,
                     sorted_weights_b, sorted_expert_ids_b, num_tokens_post_padded)
    print(f'{moe_buf=}')
    avg_a = avg_b = 1
    print(
        f"[perf] {token=}, {model_dim=}, {hidden_dim=}, {E=}, {topk=}, dtype: {dtype}, torch avg: {avg_a:.2f} us, ck avg: {avg_b:.2f} us, uplift: {avg_a/avg_b-1:.1%}", end=' ')


print('test test_fmoe')
for dtype in [torch.float16, torch.bfloat16][1:]:
    for m in [1, 2, 4, 8, 16, 32, 64, 128, 256][5:5+1]:
        for dim in [4096, 8192, 16384, 32768, 65536][1:1+1]:
            for hdim in [1024, 4096, 8192, 16384, 32768, 65536][:1]:
                test_fmoe(dtype, m, dim, hdim, 32, 5)
