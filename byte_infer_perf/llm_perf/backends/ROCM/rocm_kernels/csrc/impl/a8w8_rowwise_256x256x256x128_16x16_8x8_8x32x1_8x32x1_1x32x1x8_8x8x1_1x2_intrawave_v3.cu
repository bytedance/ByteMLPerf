// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_common.cuh"

template <typename DDataType, typename EDataType = DDataType>
torch::Tensor
a8w8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y)
{
    // A kernel that seems to work well on mid sized tensors.

    // Check if this input needs to be padded.
    int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    int N = WQ.size(0);
    int K = WQ.size(1);
    bool pad = (K % 128 != 0);

    // Dispatch based on whether padding is needed or not.
    if (pad)
    {
        using DeviceGemmInstance = DeviceGemmHelper<
            DDataType, EDataType,
            256,
            256,
            256,
            128,
            16,
            16,
            8,
            8,
            S<8, 32, 1>,
            S<8, 32, 1>,
            S<1, 32, 1, 8>,
            S<8, 8, 1>,
            1,
            2,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
            ck::tensor_operation::device::GemmSpecialization::KPadding>;
        // Run kernel instance.
        return gemm_a8w8_rowwise_impl<DDataType, EDataType, DeviceGemmInstance>(
            XQ, WQ, x_scale, w_scale, Y);
    }
    else
    {
        using DeviceGemmInstance = DeviceGemmHelper<
            DDataType, EDataType,
            256,
            256,
            256,
            128,
            16,
            16,
            8,
            8,
            S<8, 32, 1>,
            S<8, 32, 1>,
            S<1, 32, 1, 8>,
            S<8, 8, 1>,
            1,
            2,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
            ck::tensor_operation::device::GemmSpecialization::Default>;
        // Run kernel instance.
        return gemm_a8w8_rowwise_impl<DDataType, EDataType, DeviceGemmInstance>(
            XQ, WQ, x_scale, w_scale, Y);
    }
}

template torch::Tensor
a8w8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<F16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template torch::Tensor
a8w8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<B16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template torch::Tensor
a8w8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<F32, F16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template torch::Tensor
a8w8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<F32, B16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);
