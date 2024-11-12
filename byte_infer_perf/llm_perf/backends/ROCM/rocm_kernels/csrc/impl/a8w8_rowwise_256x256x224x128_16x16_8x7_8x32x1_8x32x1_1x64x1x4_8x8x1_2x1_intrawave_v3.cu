// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_common.cuh"

template <typename DDataType, typename EDataType = DDataType>
torch::Tensor
a8w8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias)
{
    // Check if this input needs to be padded.
    int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    int N = WQ.size(0);
    int K = WQ.size(1);
    bool pad = (M % 256 != 0) || (N % 224 != 0) || (K % 128 != 0);

    // This kernel seems optimal in the most purely compute bound tasks.
    if (pad)
    {
        if (bias != std::nullopt)
        {
            using DeviceGemmInstance = DeviceGemmHelperMMA<
                DDataType, EDataType,
                256,
                256,
                224,
                128,
                16,
                16,
                8,
                7,
                S<8, 32, 1>,
                S<8, 32, 1>,
                S<1, 64, 1, 4>,
                S<8, 8, 1, 1>,
                2,
                1,
                ck::BlockGemmPipelineScheduler::Intrawave,
                ck::BlockGemmPipelineVersion::v3,
                ck::tensor_operation::device::GemmSpecialization::MNPadding>;
            // Run kernel instance.
            return gemm_a8w8_mma_impl<DDataType, EDataType, DeviceGemmInstance>(
                XQ, WQ, x_scale, w_scale, Y, bias);
        }
        else
        {
            using DeviceGemmInstance = DeviceGemmHelper<
                DDataType, EDataType,
                256,
                256,
                224,
                128,
                16,
                16,
                8,
                7,
                S<8, 32, 1>,
                S<8, 32, 1>,
                S<1, 64, 1, 4>,
                S<8, 8, 1>,
                2,
                1,
                ck::BlockGemmPipelineScheduler::Intrawave,
                ck::BlockGemmPipelineVersion::v3,
                ck::tensor_operation::device::GemmSpecialization::MNPadding>;
            // Run kernel instance.
            return gemm_a8w8_rowwise_impl<DDataType, EDataType, DeviceGemmInstance>(
                XQ, WQ, x_scale, w_scale, Y, bias);
        }
    }
    else
    {
        if (bias != std::nullopt)
        {
            using DeviceGemmInstance = DeviceGemmHelperMMA<
                DDataType, EDataType,
                256,
                256,
                224,
                128,
                16,
                16,
                8,
                7,
                S<8, 32, 1>,
                S<8, 32, 1>,
                S<1, 64, 1, 4>,
                S<8, 8, 1, 1>,
                2,
                1,
                ck::BlockGemmPipelineScheduler::Intrawave,
                ck::BlockGemmPipelineVersion::v3,
                ck::tensor_operation::device::GemmSpecialization::Default>;
            // Run kernel instance.
            return gemm_a8w8_mma_impl<DDataType, EDataType, DeviceGemmInstance>(
                XQ, WQ, x_scale, w_scale, Y, bias);
        }
        else
        {
            using DeviceGemmInstance = DeviceGemmHelper<
                DDataType, EDataType,
                256,
                256,
                224,
                128,
                16,
                16,
                8,
                7,
                S<8, 32, 1>,
                S<8, 32, 1>,
                S<1, 64, 1, 4>,
                S<8, 8, 1>,
                2,
                1,
                ck::BlockGemmPipelineScheduler::Intrawave,
                ck::BlockGemmPipelineVersion::v3,
                ck::tensor_operation::device::GemmSpecialization::Default>;
            // Run kernel instance.
            return gemm_a8w8_rowwise_impl<DDataType, EDataType, DeviceGemmInstance>(
                XQ, WQ, x_scale, w_scale, Y, bias);
        }
    }
}

template torch::Tensor
a8w8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<F16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);

template torch::Tensor
a8w8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<B16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);

template torch::Tensor
a8w8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<F32, F16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);

template torch::Tensor
a8w8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<F32, B16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);