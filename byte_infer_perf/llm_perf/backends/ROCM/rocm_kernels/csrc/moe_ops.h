#pragma once

#include <torch/extension.h>

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output);

void moe_align_block_size(torch::Tensor topk_ids, int num_experts,
                          int block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad);
void silu_and_mul(torch::Tensor &out, torch::Tensor &input);

void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon);