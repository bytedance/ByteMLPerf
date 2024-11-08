import torch
import rocmKernels
BLOCK_SIZE_M = 32


def moe_sorting_ck(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype):
    block_size = BLOCK_SIZE_M
    device = topk_ids.device
    topk = topk_ids.shape[1]
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = int(
        (max_num_tokens_padded+block_size-1)//block_size)
    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=device)
    sorted_weights = torch.empty((max_num_tokens_padded, ),
                                 dtype=torch.int32,
                                 device=device)
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


def asm_moe(hidden_states, w1, w2, topk_weight, topk_ids):
    E, _, model_dim = w1.shape
    dtype = w1.dtype
    sorted_ids_b, sorted_weights_b, sorted_expert_ids_b, num_tokens_post_padded, moe_buf = moe_sorting_ck(topk_ids, topk_weight, E,
                                                                                                          model_dim, dtype)
    rocmKernels.fmoe(moe_buf, hidden_states, w1, w2, sorted_ids_b,
                     sorted_weights_b, sorted_expert_ids_b, num_tokens_post_padded)
    return moe_buf
