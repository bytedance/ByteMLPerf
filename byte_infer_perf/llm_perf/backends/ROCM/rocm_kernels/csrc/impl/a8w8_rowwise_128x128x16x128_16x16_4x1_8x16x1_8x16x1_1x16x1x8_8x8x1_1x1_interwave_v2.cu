// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_common.cuh"

template <typename DDataType, typename EDataType = DDataType>
torch::Tensor
a8w8_rowwise_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias)
{
    // A kernel that works well on small but not super tiny shapes.
    if (bias != std::nullopt)
    {
        using DeviceGemmInstance = DeviceGemmHelperMMA<
            DDataType, EDataType,
            128,
            128,
            16,
            128,
            16,
            16,
            4,
            1,
            S<8, 16, 1>,
            S<8, 16, 1>,
            S<1, 16, 1, 8>,
            S<2, 2, 1, 1>,
            1,
            1,
            ck::BlockGemmPipelineScheduler::Interwave,
            ck::BlockGemmPipelineVersion::v2>;
        // Run kernel instance.
        return gemm_a8w8_mma_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias);
    }
    else
    {
        using DeviceGemmInstance = DeviceGemmHelper<
            DDataType, EDataType,
            128,
            128,
            16,
            128,
            16,
            16,
            4,
            1,
            S<8, 16, 1>,
            S<8, 16, 1>,
            S<1, 16, 1, 8>,
            S<2, 2, 1>,
            1,
            1,
            ck::BlockGemmPipelineScheduler::Interwave,
            ck::BlockGemmPipelineVersion::v2>;
        // Run kernel instance.
        return gemm_a8w8_rowwise_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias);
    }
}

template torch::Tensor
a8w8_rowwise_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2<F16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);

template torch::Tensor
a8w8_rowwise_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2<B16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);

template torch::Tensor
a8w8_rowwise_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2<F32, F16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);

template torch::Tensor
a8w8_rowwise_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2<F32, B16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);