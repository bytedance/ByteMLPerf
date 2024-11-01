/*
 * Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
 *
 * @Script: ck_py_interface.cpp
 * @Author: valarLip
 * @Email: lingpeng.jin@amd.com
 * @Create At: 2024-10-24 12:11:58
 * @Last Modified By: valarLip
 * @Last Modified At: 2024-11-01 00:36:13
 * @Description: This is description.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(FIND_CK)
#include "layernorm2d_fwd.hpp"

// common utility functions
#define FOREACH_BUFFER_TORCH_TYPE_MAP(F) \
    F("fp32", torch::kFloat)             \
    F("fp16", torch::kHalf)              \
    F("bf16", torch::kBFloat16)

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
        throw std::runtime_error("Unsupported data type " + std::to_string((int8_t)(dtype.toScalarType())));
    }

#undef TYPE_CASE
}

void layernorm2d(torch::Tensor &out,    // [hidden_size]
                 torch::Tensor &input,  // [hidden_size]
                 torch::Tensor &weight, // [hidden_size]
                 torch::Tensor &bias,   // [hidden_size]
                 double epsilon)
{
    auto dtype = input.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "ck layernorm2d only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(dtype);
    int n = input.size(-1);
    int m = input.numel() / n;
    int stride = n;
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
                     static_cast<float>(epsilon), m, n, stride},
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
    int stride = n;
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
                     static_cast<float>(epsilon), m, n, stride},
                    {stream});
}
#endif //
