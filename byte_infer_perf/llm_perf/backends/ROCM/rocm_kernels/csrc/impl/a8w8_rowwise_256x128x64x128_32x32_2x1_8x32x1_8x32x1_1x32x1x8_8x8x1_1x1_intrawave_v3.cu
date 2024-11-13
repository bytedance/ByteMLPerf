// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_common.cuh"

template <typename DDataType, typename EDataType = DDataType>
torch::Tensor
a8w8_rowwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias)
{
    // A kernel that seems to work well on mid sized tensors.

    // Check if this input needs to be padded.
    int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    int N = WQ.size(0);
    int K = WQ.size(1);
    bool pad = (M % 128 != 0) || (N % 64 != 0) || (K % 128 != 0);

    // Dispatch based on whether padding is needed or not.
    if (pad)
    {
        if (bias != std::nullopt)
        {
            using DeviceGemmInstance = DeviceGemmHelperMMA<
                DDataType, EDataType,
                256,
                128,
                64,
                128,
                32,
                32,
                2,
                1,
                S<8, 32, 1>,
                S<8, 32, 1>,
                S<1, 32, 1, 8>,
                S<8, 8, 1, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Intrawave,
                ck::BlockGemmPipelineVersion::v3,
                ck::tensor_operation::device::GemmSpecialization::MNPadding>;
            // Run kernel instance.
            return gemm_a8w8_mma_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias);
        }
        else
        {
            using DeviceGemmInstance = DeviceGemmHelper<
                DDataType, EDataType,
                256,
                128,
                64,
                128,
                32,
                32,
                2,
                1,
                S<8, 32, 1>,
                S<8, 32, 1>,
                S<1, 32, 1, 8>,
                S<8, 8, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Intrawave,
                ck::BlockGemmPipelineVersion::v3,
                ck::tensor_operation::device::GemmSpecialization::MNPadding>;
            // Run kernel instance.
            return gemm_a8w8_rowwise_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias);
        }
    }
    else
    {
        if (bias != std::nullopt)
        {
            using DeviceGemmInstance = DeviceGemmHelperMMA<
                DDataType, EDataType,
                256,
                128,
                64,
                128,
                32,
                32,
                2,
                1,
                S<8, 32, 1>,
                S<8, 32, 1>,
                S<1, 32, 1, 8>,
                S<8, 8, 1, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Intrawave,
                ck::BlockGemmPipelineVersion::v3,
                ck::tensor_operation::device::GemmSpecialization::Default>;
            // Run kernel instance.
            return gemm_a8w8_mma_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias);
        }
        else
        {
            using DeviceGemmInstance = DeviceGemmHelper<
                DDataType, EDataType,
                256,
                128,
                64,
                128,
                32,
                32,
                2,
                1,
                S<8, 32, 1>,
                S<8, 32, 1>,
                S<1, 32, 1, 8>,
                S<8, 8, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Intrawave,
                ck::BlockGemmPipelineVersion::v3,
                ck::tensor_operation::device::GemmSpecialization::Default>;
            // Run kernel instance.
            return gemm_a8w8_rowwise_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias);
        }
    }
}

template torch::Tensor
a8w8_rowwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<F16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);

template torch::Tensor
a8w8_rowwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<B16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);

template torch::Tensor
a8w8_rowwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<F32, F16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);

template torch::Tensor
a8w8_rowwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<F32, B16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);