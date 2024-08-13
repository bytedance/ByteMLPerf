
#include "ixgemmblaslt.hpp"


gemm_kernel_param gemm_init()
{
  return gemm_kernel_init();
}

void free_device(void * d_data)
{
  cudaFree(d_data);
}

std::vector<at::Tensor> gemm_run(gemm_kernel_param pins, std::vector<at::Tensor> &alist, std::vector<at::Tensor> &blist)
{
  std::vector<at::Tensor> clist(alist.size());

  for (size_t i = 0; i < alist.size(); i++)
  {
    //int dataSize = alist[i].numel();
    c10::IntArrayRef shape_a = alist[i].sizes();
    c10::IntArrayRef shape_b = blist[i].sizes();
    if(shape_a.size() == 2 && shape_b.size() == 2)
    {
      // 二维矩阵
      {
        int *d_c;
        cudaMalloc((void **)&d_c, sizeof(int) * shape_a[0] * shape_b[1]);
        auto options = torch::TensorOptions().dtype(torch::kInt32);
        options.device(at::kCUDA);
        clist[i] = torch::from_blob(d_c, {shape_a[0], shape_b[1]}, std::bind(&free_device, d_c), options);
        clist[i] = clist[i].cuda();
      }
      int M = shape_a[0];
      int N = shape_b[1];
      int K = shape_a[1];
      gemm_kernel_run(pins, (char *)alist[i].data_ptr(), (char *)blist[i].data_ptr(), (int32_t *)clist[i].data_ptr(), M, N, K);
    }
    else if (shape_a.size() == 3 && shape_b.size() == 3)
    {
      // 三维矩阵
      {
        int *d_c;
        cudaMalloc((void **)&d_c, sizeof(int) * shape_a[0] * shape_a[1] * shape_b[2]);
        auto options = torch::TensorOptions().dtype(torch::kInt32);
        options.device(at::kCUDA);
        clist[i] = torch::from_blob(d_c, {shape_a[0], shape_a[1], shape_b[2]}, std::bind(&free_device, d_c), options);
        clist[i] = clist[i].cuda();
      }
      for (size_t j = 0; j < shape_a[0]; j++)
      {
        int M = shape_a[1];
        int N = shape_b[2];
        int K = shape_a[2];
        gemm_kernel_run(pins, (char *)alist[i][j].data_ptr(), (char *)blist[i][j].data_ptr(), (int32_t *)clist[i][j].data_ptr(), M, N, K);
      }
    }
    else
    {
      std::cout << "tensor shapes are illegal" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  cudaDeviceSynchronize();
  return clist;
}

void gemm_release(gemm_kernel_param ins)
{
  gemm_kernel_release(ins);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<gemm_kernel_param>(m, "gemm_kernel_param")
        .def_readwrite("lt_handle", &gemm_kernel_param::lt_handle)
        .def_readwrite("op_desc", &gemm_kernel_param::op_desc);
  m.def("gemm_init", &gemm_init, "");
  m.def("gemm_run", &gemm_run, "");
  m.def("gemm_release", &gemm_release, "");
}

