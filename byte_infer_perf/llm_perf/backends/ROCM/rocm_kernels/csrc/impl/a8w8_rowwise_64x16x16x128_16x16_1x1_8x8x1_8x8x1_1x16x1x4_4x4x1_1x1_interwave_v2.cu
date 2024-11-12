// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_common.cuh"

template <typename DDataType, typename EDataType = DDataType>
torch::Tensor
a8w8_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias)
{
    // The smallest kernel we have available. Works well for memory bound shapes.

    // Check if this input needs to be padded.
    int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    int N = WQ.size(0);
    int K = WQ.size(1);
    bool pad = (M % 16 != 0) || (N % 16 != 0) || (K % 128 != 0);
    if (pad)
    {
        if (bias != std::nullopt)
        {
            using DeviceGemmInstance = DeviceGemmHelperMMA<
                DDataType, EDataType,
                64,
                16,
                16,
                128,
                16,
                16,
                1,
                1,
                S<8, 8, 1>,
                S<8, 8, 1>,
                S<1, 16, 1, 4>,
                S<4, 4, 1, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Interwave,
                ck::BlockGemmPipelineVersion::v2,
                ck::tensor_operation::device::GemmSpecialization::MNPadding>;
            // Run kernel instance.
            return gemm_a8w8_mma_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias);
        }
        else
        {
            using DeviceGemmInstance = DeviceGemmHelper<
                DDataType, EDataType,
                64,
                16,
                16,
                128,
                16,
                16,
                1,
                1,
                S<8, 8, 1>,
                S<8, 8, 1>,
                S<1, 16, 1, 4>,
                S<4, 4, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Interwave,
                ck::BlockGemmPipelineVersion::v2,
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
                64,
                16,
                16,
                128,
                16,
                16,
                1,
                1,
                S<8, 8, 1>,
                S<8, 8, 1>,
                S<1, 16, 1, 4>,
                S<4, 4, 1, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Interwave,
                ck::BlockGemmPipelineVersion::v2,
                ck::tensor_operation::device::GemmSpecialization::Default>;
            // Run kernel instance.
            return gemm_a8w8_mma_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias);
        }
        else
        {
            using DeviceGemmInstance = DeviceGemmHelper<
                DDataType, EDataType,
                64,
                16,
                16,
                128,
                16,
                16,
                1,
                1,
                S<8, 8, 1>,
                S<8, 8, 1>,
                S<1, 16, 1, 4>,
                S<4, 4, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Interwave,
                ck::BlockGemmPipelineVersion::v2,
                ck::tensor_operation::device::GemmSpecialization::Default>;
            // Run kernel instance.
            return gemm_a8w8_rowwise_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias);
        }
    }
}

template torch::Tensor
a8w8_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2<F16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);

template torch::Tensor
a8w8_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2<B16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);

template torch::Tensor
a8w8_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2<F32, F16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);

template torch::Tensor
a8w8_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2<F32, B16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);
