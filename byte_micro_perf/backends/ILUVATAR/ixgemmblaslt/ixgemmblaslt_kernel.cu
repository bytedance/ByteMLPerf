#include <iostream>
#include "ixgemmblaslt.hpp"

//#define CAL_TFLOPS_TEST

gemm_kernel_param gemm_kernel_init()
{
  cublasLtHandle_t lt_handle = nullptr;
  checkBlasStatus(cublasLtCreate(&(lt_handle)));

  cublasLtMatmulDesc_t op_desc = nullptr;
#ifdef __ILUVATAR__
    cudaDataType compute_type = CUDA_R_32I;
#else
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32I;
#endif
  //cudaDataType scale_type = CUDA_R_32I;
  cudaDataType scale_type = CUDA_R_32F;
  cublasOperation_t op_trans_a = CUBLAS_OP_N;
  cublasOperation_t op_trans_b = CUBLAS_OP_N;
#ifdef __ILUVATAR__
    checkBlasStatus(cublasLtMatmulDescCreate(&op_desc, compute_type));
#else
    checkBlasStatus(cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type));
#endif
  checkBlasStatus(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));
  checkBlasStatus(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_trans_a, sizeof(op_trans_a)));
  checkBlasStatus(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_trans_b, sizeof(op_trans_b)));

  gemm_kernel_param ins;
  ins.lt_handle = reinterpret_cast<uintptr_t>(lt_handle);
  ins.op_desc = reinterpret_cast<uintptr_t>(op_desc);

  return ins;
}

void gemm_kernel_run(gemm_kernel_param ins, char *d_A, char *d_B, char *d_C, const int M, const int N, const int K)
{
  float alpha_int8 = 1.0;
  float beta_int8 = 0.0;
  cudaDataType ab_type = CUDA_R_8I;
  //cudaDataType c_type = CUDA_R_32I;
  cudaDataType c_type = CUDA_R_8I;
  cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr;

  cublasLtHandle_t lt_handle = reinterpret_cast<cublasLtHandle_t>(reinterpret_cast<uintptr_t *>(ins.lt_handle));
  cublasLtMatmulDesc_t op_desc = reinterpret_cast<cublasLtMatmulDesc_t>(reinterpret_cast<uintptr_t *>(ins.op_desc));

  checkBlasStatus(cublasLtMatrixLayoutCreate(&a_desc, ab_type, K, M, K));
  checkBlasStatus(cublasLtMatrixLayoutCreate(&b_desc, ab_type, N, K, N));
  checkBlasStatus(cublasLtMatrixLayoutCreate(&c_desc, c_type, N, M, N));

#ifdef CAL_TFLOPS_TEST
  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
#endif

  checkBlasStatus(cublasLtMatmul(lt_handle, op_desc, &alpha_int8, d_B, b_desc, d_A, a_desc, &beta_int8, d_C, c_desc, d_C, c_desc, nullptr, nullptr, 0, nullptr));

#ifdef CAL_TFLOPS_TEST  
  cudaThreadSynchronize();
  auto stop = std::chrono::steady_clock::now();

  std::chrono::duration<double, std::milli> dur_ms = stop - start;
  double elapse = dur_ms.count();
  double tflops = 1e-9 * 2.0f * M * N * K;
  printf("\n---------------elapse: %lf ms, TOPs: %lf\n\n", elapse, tflops / elapse);
#endif
}

void gemm_kernel_release(gemm_kernel_param ins)
{
  cublasLtHandle_t lt_handle = reinterpret_cast<cublasLtHandle_t>(reinterpret_cast<uintptr_t *>(ins.lt_handle));
  cublasLtDestroy(lt_handle);
}
