#pragma once

#include <torch/extension.h>

void topk_softmax(torch::Tensor &topk_weights, torch::Tensor &topk_indices,
                  torch::Tensor &token_expert_indices,
                  torch::Tensor &gating_output);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad);

void silu_and_mul(torch::Tensor &out, torch::Tensor &input);

void rms_norm(torch::Tensor &out, torch::Tensor &input, torch::Tensor &weight,
              double epsilon);

void fused_add_rms_norm(torch::Tensor &input, torch::Tensor &residual,
                        torch::Tensor &weight, double epsilon);

void wvSpltK(at::Tensor &in_a, at::Tensor &in_b, at::Tensor &out_c,
             const int64_t N_in, const int64_t CuCount);

void rotary_embedding(torch::Tensor &positions, torch::Tensor &query,
                      torch::Tensor &key, int64_t head_size,
                      torch::Tensor &cos_sin_cache, bool is_neox);

void batched_rotary_embedding(torch::Tensor &positions, torch::Tensor &query,
                              torch::Tensor &key, int64_t head_size,
                              torch::Tensor &cos_sin_cache, bool is_neox,
                              int64_t rot_dim,
                              torch::Tensor &cos_sin_cache_offsets);
