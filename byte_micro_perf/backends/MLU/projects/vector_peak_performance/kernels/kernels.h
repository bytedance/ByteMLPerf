#ifndef TOOLS_CAMBRICON_BANGC_BENCHMARK_TOOLS_CAMBRICON_PEAK_PERFORMANCE_BENCHMARK_TOOL_TEST_FMA_H_
#define TOOLS_CAMBRICON_BANGC_BENCHMARK_TOOLS_CAMBRICON_PEAK_PERFORMANCE_BENCHMARK_TOOL_TEST_FMA_H_

#define REPEAT 1000
#define NUM 32768

struct PeakTimeInfo {
    uint32_t fusion_hardware_time;
    float fusion_tflops;
};

#if defined(__BANG__)
#include <mlu.h>
#endif  // defined(__BANG__)

#ifdef __BANG__
#define MAX_WRAM_SIZE (__MLU_WRAM_SIZE__ * 1024)
#define WRAM_LT_STRIDE (__MLU_WRAM_SIZE__ * 1024 / 64)
#define REM_FOR_STACK (128 * 1024)           // 128KB reserved for cncc
#define MAX_NRAM_SIZE (__MLU_NRAM_SIZE__ * 1024 - REM_FOR_STACK)  // 128KB reserved for cncc
#else
#define MAX_NRAM_SIZE (384 * 1024)           // 384KB, initialization value
#endif

#ifndef __BANG__
#ifdef __cplusplus
extern "C" {
#endif
void fma_float32(PeakTimeInfo *peak_ptr);
void fma_float16(PeakTimeInfo *peak_ptr);
void fma_bfloat16(PeakTimeInfo *peak_ptr);
void fma_int8(PeakTimeInfo *peak_ptr);

void pow2_float32(PeakTimeInfo *peak_ptr);
void pow2_bfloat16(PeakTimeInfo *peak_ptr);
#ifdef __cplusplus
}
#endif
#else
#ifdef __cplusplus
extern "C" {
#endif
// fma
__mlu_global__ void fma_float32(PeakTimeInfo *peak_ptr);
__mlu_global__ void fma_float16(PeakTimeInfo *peak_ptr);
__mlu_global__ void fma_bfloat16(PeakTimeInfo *peak_ptr);
__mlu_global__ void fma_int8(PeakTimeInfo *peak_ptr);

// pow2
__mlu_global__ void pow2_float32(PeakTimeInfo *peak_ptr);
__mlu_global__ void pow2_bfloat16(PeakTimeInfo *peak_ptr);

#ifdef __cplusplus
}
#endif
#endif

#endif  // TOOLS_CAMBRICON_BANGC_BENCHMARK_TOOLS_CAMBRICON_PEAK_PERFORMANCE_BENCHMARK_TOOL_TEST_CONV_H_
