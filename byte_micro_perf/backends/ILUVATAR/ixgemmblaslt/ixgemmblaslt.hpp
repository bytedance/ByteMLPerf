#ifndef IXGEMMBLASLT_HPP
#define IXGEMMBLASLT_HPP

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <torch/extension.h>
#include <pybind11/stl.h>

#define checkBlasStatus(status)                                               \
  do                                                                          \
  {                                                                           \
    if (status != CUBLAS_STATUS_SUCCESS)                                      \
    {                                                                         \
      std::cout << "cublasLt API failed with status " << status << std::endl; \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

struct gemm_kernel_param
{
    gemm_kernel_param(){

    }
  uintptr_t lt_handle;
  uintptr_t op_desc;
};


gemm_kernel_param gemm_kernel_init();

void gemm_kernel_run(gemm_kernel_param pins, char *d_A, char *d_B, char *d_C, const int M, const int N, const int K);

void gemm_kernel_release(gemm_kernel_param pins);

#endif // !IXGEMMBLASLT_HPP
