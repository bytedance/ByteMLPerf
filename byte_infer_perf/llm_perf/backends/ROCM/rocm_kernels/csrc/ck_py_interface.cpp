/*
 * Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
 *
 * @Script: ck_py_interface.cpp
 * @Author: valarLip
 * @Email: lingpeng.jin@amd.com
 * @Create At: 2024-10-24 12:11:58
 * @Last Modified By: valarLip
 * @Last Modified At: 2024-11-10 17:52:30
 * @Description: This is description.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(FIND_CK)
#include "layernorm2d_fwd.hpp"
#include "smoothquant.hpp"
#include "moe_sorting_api.hpp"

// common utility functions
#define FOREACH_BUFFER_TORCH_TYPE_MAP(F) \
    F("fp32", torch::kFloat)             \
    F("fp16", torch::kHalf)              \
    F("bf16", torch::kBFloat16)          \
    F("int32", torch::kInt32)            \
    F("int8", torch::kInt8)

inline std::string torchDTypeToStr(caffe2::TypeMeta dtype)
{
#define TYPE_CASE(type, torch_type) \
    case torch_type:                \
    {                               \
        return type;                \
    }

    switch (dtype.toScalarType())
    {
        FOREACH_BUFFER_TORCH_TYPE_MAP(TYPE_CASE);
    default:
        throw std::runtime_error("CKPyInterface: Unsupported data type " + std::to_string((int8_t)(dtype.toScalarType())));
    }

#undef TYPE_CASE
}

void layernorm2d(torch::Tensor &out,    // [m, n]
                 torch::Tensor &input,  // [m, n]
                 torch::Tensor &weight, // [m, n]
                 torch::Tensor &bias,   // [m, n]
                 double epsilon)
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "ck layernorm2d only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(dtype);
    int n = input.size(-1);
    int m = input.numel() / n;
    int stride = input.stride(0);
    int xr_stride = -1;
    int y_stride = out.stride(0);
    int yr_stride = -1;
    bool SaveMeanVar = false;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    layernorm2d_fwd({
                        dtype_str, // input precision
                        dtype_str, // output precision
                        dtype_str, // x-scale, used for [1*N] input smooth quant
                        dtype_str, // y-scale, used for [M*1] output for next layer
                        SaveMeanVar,
                        0, // fused_add
                        0  // fused_quant
                    },
                    {input.data_ptr(),
                     nullptr, // p_x_residual
                     nullptr, // p_x_scale
                     weight.data_ptr(), bias.data_ptr(), out.data_ptr(),
                     nullptr, // p_y_residual
                     nullptr, // p_y_scale
                     nullptr, // p_mean
                     nullptr, // p_invStd
                     static_cast<float>(epsilon), m, n, stride, xr_stride, y_stride, yr_stride},
                    {stream});
}

void layernorm2d_with_add(torch::Tensor &out,          // [m ,n]
                          torch::Tensor &input,        // [m ,n]
                          torch::Tensor &residual_in,  // [m ,n]
                          torch::Tensor &residual_out, // [m ,n]
                          torch::Tensor &weight,       // [1 ,n]
                          torch::Tensor &bias,         // [1 ,n]
                          double epsilon)
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "ck layernorm2d only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(input.dtype());
    int n = input.size(-1);
    int m = input.numel() / n;
    int stride = input.stride(0);
    int xr_stride = residual_in.stride(0);
    int y_stride = out.stride(0);
    int yr_stride = residual_out.stride(0);
    bool SaveMeanVar = false;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    layernorm2d_fwd({
                        dtype_str, // input precision
                        dtype_str, // output precision
                        dtype_str, // x-scale, used for [1*N] input smooth quant
                        dtype_str, // y-scale, used for [M*1] output for next layer
                        SaveMeanVar,
                        1, // fused_add
                        0  // fused_quant
                    },
                    {input.data_ptr(),       // p_x
                     residual_in.data_ptr(), // p_x_residual
                     nullptr,                // p_x_scale
                     weight.data_ptr(),      // p_gamma
                     bias.data_ptr(),        // p_beta

                     out.data_ptr(),          // p_y
                     residual_out.data_ptr(), // p_y_residual
                     nullptr,                 // p_y_scale
                     nullptr,                 // p_mean
                     nullptr,                 // p_invStd
                     static_cast<float>(epsilon), m, n, stride, xr_stride, y_stride, yr_stride},
                    {stream});
}

void layernorm2d_with_smoothquant(torch::Tensor &out,    // [m ,n]
                                  torch::Tensor &input,  // [m ,n]
                                  torch::Tensor &xscale, // [1 ,n]
                                  torch::Tensor &yscale, // [m ,1]
                                  torch::Tensor &weight, // [1 ,n]
                                  torch::Tensor &bias,   // [1 ,n]
                                  double epsilon)
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "ck layernorm2d only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(input.dtype());
    std::string out_dtype_str = torchDTypeToStr(out.dtype());
    std::string xscale_dtype_str = torchDTypeToStr(xscale.dtype());
    std::string yscale_dtype_str = torchDTypeToStr(yscale.dtype());
    int n = input.size(-1);
    int m = input.numel() / n;
    int stride = input.stride(0);
    int xr_stride = -1;
    int y_stride = out.stride(0);
    int yr_stride = -1;
    bool SaveMeanVar = false;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    layernorm2d_fwd({
                        dtype_str,        // input precision
                        out_dtype_str,    // output precision
                        xscale_dtype_str, // x-scale, used for [1*N] input smooth quant
                        yscale_dtype_str, // y-scale, used for [M*1] output for next layer
                        SaveMeanVar,
                        0, // fused_add
                        1  // fused_quant
                    },
                    {input.data_ptr(),  // p_x
                     nullptr,           // p_x_residual
                     xscale.data_ptr(), // p_x_scale
                     weight.data_ptr(), // p_gamma
                     bias.data_ptr(),   // p_beta

                     out.data_ptr(),    // p_y
                     nullptr,           // p_y_residual
                     yscale.data_ptr(), // p_y_scale
                     nullptr,           // p_mean
                     nullptr,           // p_invStd
                     static_cast<float>(epsilon), m, n, stride, xr_stride, y_stride, yr_stride},
                    {stream});
}

void layernorm2d_with_add_smoothquant(torch::Tensor &out,          // [m ,n]
                                      torch::Tensor &input,        // [m ,n]
                                      torch::Tensor &residual_in,  // [m ,n]
                                      torch::Tensor &residual_out, // [m ,n]
                                      torch::Tensor &xscale,       // [1 ,n]
                                      torch::Tensor &yscale,       // [m ,1]
                                      torch::Tensor &weight,       // [1 ,n]
                                      torch::Tensor &bias,         // [1 ,n]
                                      double epsilon)
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "ck layernorm2d only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(input.dtype());
    std::string out_dtype_str = torchDTypeToStr(out.dtype());
    std::string xscale_dtype_str = torchDTypeToStr(xscale.dtype());
    std::string yscale_dtype_str = torchDTypeToStr(yscale.dtype());
    int n = input.size(-1);
    int m = input.numel() / n;
    int stride = input.stride(0);
    int xr_stride = residual_in.stride(0);
    int y_stride = out.stride(0);
    int yr_stride = residual_out.stride(0);
    bool SaveMeanVar = false;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    layernorm2d_fwd({
                        dtype_str,        // input precision
                        out_dtype_str,    // output precision
                        xscale_dtype_str, // x-scale, used for [1*N] input smooth quant
                        yscale_dtype_str, // y-scale, used for [M*1] output for next layer
                        SaveMeanVar,
                        1, // fused_add
                        1  // fused_quant
                    },
                    {input.data_ptr(),       // p_x
                     residual_in.data_ptr(), // p_x_residual
                     xscale.data_ptr(),      // p_x_scale
                     weight.data_ptr(),      // p_gamma
                     bias.data_ptr(),        // p_beta

                     out.data_ptr(),          // p_y
                     residual_out.data_ptr(), // p_y_residual
                     yscale.data_ptr(),       // p_y_scale
                     nullptr,                 // p_mean
                     nullptr,                 // p_invStd
                     static_cast<float>(epsilon), m, n, stride, xr_stride, y_stride, yr_stride},
                    {stream});
}

void layernorm2d_with_dynamicquant(torch::Tensor &out,    // [m ,n]
                                   torch::Tensor &input,  // [m ,n]
                                   torch::Tensor &yscale, // [m ,1]
                                   torch::Tensor &weight, // [1 ,n]
                                   torch::Tensor &bias,   // [1 ,n]
                                   double epsilon)
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "ck layernorm2d only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(input.dtype());
    std::string out_dtype_str = torchDTypeToStr(out.dtype());
    std::string yscale_dtype_str = torchDTypeToStr(yscale.dtype());
    int n = input.size(-1);
    int m = input.numel() / n;
    int stride = input.stride(0);
    int xr_stride = -1;
    int y_stride = out.stride(0);
    int yr_stride = -1;
    bool SaveMeanVar = false;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    layernorm2d_fwd({
                        dtype_str,        // input precision
                        out_dtype_str,    // output precision
                        dtype_str,        // x-scale, used for [1*N] input smooth quant
                        yscale_dtype_str, // y-scale, used for [M*1] output for next layer
                        SaveMeanVar,
                        0, // fused_add
                        2  // fused_quant
                    },
                    {input.data_ptr(),  // p_x
                     nullptr,           // p_x_residual
                     nullptr,           // p_x_scale
                     weight.data_ptr(), // p_gamma
                     bias.data_ptr(),   // p_beta

                     out.data_ptr(),    // p_y
                     nullptr,           // p_y_residual
                     yscale.data_ptr(), // p_y_scale
                     nullptr,           // p_mean
                     nullptr,           // p_invStd
                     static_cast<float>(epsilon), m, n, stride, xr_stride, y_stride, yr_stride},
                    {stream});
}

void layernorm2d_with_add_dynamicquant(torch::Tensor &out,          // [m ,n]
                                       torch::Tensor &input,        // [m ,n]
                                       torch::Tensor &residual_in,  // [m ,n]
                                       torch::Tensor &residual_out, // [m ,n]
                                       torch::Tensor &yscale,       // [m ,1]
                                       torch::Tensor &weight,       // [1 ,n]
                                       torch::Tensor &bias,         // [1 ,n]
                                       double epsilon)
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "ck layernorm2d only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(input.dtype());
    std::string out_dtype_str = torchDTypeToStr(out.dtype());
    std::string yscale_dtype_str = torchDTypeToStr(yscale.dtype());
    int n = input.size(-1);
    int m = input.numel() / n;
    int stride = input.stride(0);
    int xr_stride = residual_in.stride(0);
    int y_stride = out.stride(0);
    int yr_stride = residual_out.stride(0);
    bool SaveMeanVar = false;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    layernorm2d_fwd({
                        dtype_str,        // input precision
                        out_dtype_str,    // output precision
                        dtype_str,        // x-scale, used for [1*N] input smooth quant
                        yscale_dtype_str, // y-scale, used for [M*1] output for next layer
                        SaveMeanVar,
                        1, // fused_add
                        2  // fused_quant
                    },
                    {input.data_ptr(),       // p_x
                     residual_in.data_ptr(), // p_x_residual
                     nullptr,                // p_x_scale
                     weight.data_ptr(),      // p_gamma
                     bias.data_ptr(),        // p_beta

                     out.data_ptr(),          // p_y
                     residual_out.data_ptr(), // p_y_residual
                     yscale.data_ptr(),       // p_y_scale
                     nullptr,                 // p_mean
                     nullptr,                 // p_invStd
                     static_cast<float>(epsilon), m, n, stride, xr_stride, y_stride, yr_stride},
                    {stream});
}

void smoothquant_fwd(torch::Tensor &out,     // [m ,n]
                     torch::Tensor &input,   // [m ,n]
                     torch::Tensor &x_scale, // [1 ,n]
                     torch::Tensor &y_scale) // [m ,1]
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "ck layernorm2d only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(input.dtype());
    int n = input.size(-1);
    int m = input.numel() / n;
    int stride = n;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    smoothquant({
                    dtype_str // input  dtype
                },
                {input.data_ptr(),   // p_x
                 x_scale.data_ptr(), // p_x_scale
                 y_scale.data_ptr(), // p_y
                 out.data_ptr(),     // p_y_scale
                 m, n, stride},
                {stream});
}

void moe_sorting_fwd(torch::Tensor &topk_ids,              // [m, topk]
                     torch::Tensor &topk_weights,          // [m, topk]
                     torch::Tensor &sorted_token_ids,      // [max_num_tokens_padded]
                     torch::Tensor &sorted_weights,        // [max_num_tokens_padded]
                     torch::Tensor &sorted_expert_ids,     // [max_num_m_blocks]
                     torch::Tensor &total_tokens_post_pad, // [1]
                     torch::Tensor &moe_buf,               // [max_num_tokens_padded]
                     int num_experts,
                     int unit_size)
{
    auto dtype = topk_ids.dtype();

    auto dtype_str = torchDTypeToStr(topk_ids.dtype());
    int num_tokens = topk_ids.size(0);
    int topk = topk_ids.size(1);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    moe_sorting({
                    dtype_str, // index_type
                    "fp32"     // weight_type; // currently always float
                },
                {topk_ids.data_ptr(),              // p_topk_ids
                 topk_weights.data_ptr(),          // p_weights
                 sorted_token_ids.data_ptr(),      // p_sorted_token_ids
                 sorted_weights.data_ptr(),        // p_sorted_weights
                 sorted_expert_ids.data_ptr(),     // p_sorted_expert_ids
                 total_tokens_post_pad.data_ptr(), // p_total_tokens_post_pad
                 moe_buf.data_ptr(),               // p_moe_buf
                 num_tokens, unit_size, num_experts, topk, (int)moe_buf.nbytes()},
                {stream});
}

#endif //
