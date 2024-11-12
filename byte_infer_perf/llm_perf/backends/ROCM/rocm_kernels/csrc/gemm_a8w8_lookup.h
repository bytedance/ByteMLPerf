#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(DTYPE, ETYPE)                                                                                      \
   {                                                                                                                             \
       {{128, 1280, 8192},                                                                                                       \
        a8w8_rowwise_256x32x64x512_16x16_2x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{128, 8192, 1024},                                                                                                       \
        a8w8_rowwise_256x128x128x256_32x32_2x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                   \
       {{4096, 1280, 8192},                                                                                                      \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{4096, 8192, 1024},                                                                                                      \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>}, /* QWen-57B         \
                                                                                                            NK= 4608, 3584 */    \
       {{1, 4608, 3584},                                                                                                         \
        a8w8_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2<DTYPE, ETYPE>},                          \
       {{32, 4608, 3584},                                                                                                        \
        a8w8_rowwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2<DTYPE, ETYPE>},                       \
       {{64, 4608, 3584},                                                                                                        \
        a8w8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{128, 4608, 3584},                                                                                                       \
        a8w8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 4608, 3584},                                                                                                       \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{512, 4608, 3584},                                                                                                       \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{1024, 4608, 3584},                                                                                                      \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{2048, 4608, 3584},                                                                                                      \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{4096, 4608, 3584},                                                                                                      \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{8192, 4608, 3584},                                                                                                      \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{16384, 4608, 3584},                                                                                                     \
        a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE, ETYPE>},                      \
       {{20480, 4608, 3584},                                                                                                     \
        a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE, ETYPE>}, /* QWen-57B          \
                                                                                                             NK= 3584, 3584 */   \
       {{1, 3584, 3584},                                                                                                         \
        a8w8_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2<DTYPE, ETYPE>},                          \
       {{32, 3584, 3584},                                                                                                        \
        a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2<DTYPE, ETYPE>},                       \
       {{64, 3584, 3584},                                                                                                        \
        a8w8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{128, 3584, 3584},                                                                                                       \
        a8w8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 3584, 3584},                                                                                                       \
        a8w8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 3584, 3584},                                                                                                       \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{1024, 3584, 3584},                                                                                                      \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{2048, 3584, 3584},                                                                                                      \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{4096, 3584, 3584},                                                                                                      \
        a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE, ETYPE>},                      \
       {{8192, 3584, 3584},                                                                                                      \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{16384, 3584, 3584},                                                                                                     \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{20480, 3584, 3584},                                                                                                     \
        a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE, ETYPE>}, /* QWen-57B          \
                                                                                                             NK= 3584, 20480 */  \
       {{1, 3584, 20480},                                                                                                        \
        a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2<DTYPE, ETYPE>},                       \
       {{32, 3584, 20480},                                                                                                       \
        a8w8_rowwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2<DTYPE, ETYPE>},                       \
       {{64, 3584, 20480},                                                                                                       \
        a8w8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{128, 3584, 20480},                                                                                                      \
        a8w8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 3584, 20480},                                                                                                      \
        a8w8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 3584, 20480},                                                                                                      \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{1024, 3584, 20480},                                                                                                     \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{2048, 3584, 20480},                                                                                                     \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{4096, 3584, 20480},                                                                                                     \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{8192, 3584, 20480},                                                                                                     \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{16384, 3584, 20480},                                                                                                    \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{20480, 3584, 20480},                                                                                                    \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>}, /* QWen-57B         \
                                                                                                              NK= 40960, 3584 */ \
       {{1, 40960, 3584},                                                                                                        \
        a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2<DTYPE, ETYPE>},                       \
       {{32, 40960, 3584},                                                                                                       \
        a8w8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{64, 40960, 3584},                                                                                                       \
        a8w8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{128, 40960, 3584},                                                                                                      \
        a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE, ETYPE>},                     \
       {{256, 40960, 3584},                                                                                                      \
        a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE, ETYPE>},                      \
       {{512, 40960, 3584},                                                                                                      \
        a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE, ETYPE>},                      \
       {{1024, 40960, 3584},                                                                                                     \
        a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE, ETYPE>},                      \
       {{2048, 40960, 3584},                                                                                                     \
        a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE, ETYPE>},                      \
       {{4096, 40960, 3584},                                                                                                     \
        a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE, ETYPE>},                      \
       {{8192, 40960, 3584},                                                                                                     \
        a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE, ETYPE>},                      \
       {{16384, 40960, 3584},                                                                                                    \
        a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE, ETYPE>},                      \
       {{20480, 40960, 3584},                                                                                                    \
        a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE, ETYPE>},                      \
   }

#endif // USE_ROCM
