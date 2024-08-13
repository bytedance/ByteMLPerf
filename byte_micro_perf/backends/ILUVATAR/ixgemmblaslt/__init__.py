import os
from torch.utils.cpp_extension import load as load_cplusplus

print("to build ixgemmblaslt module ...")

cur_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(cur_dir, 'build_tmp')
if not os.path.exists(build_dir):
    os.makedirs(build_dir, exist_ok=True)

gemm88 = load_cplusplus(
    name='gemm88',
    extra_cflags=['-std=c++17',
                ],
    extra_cuda_cflags=['-std=c++17', 
                        #'-DCAL_TFLOPS_TEST',
                    ],
    sources=[os.path.join(cur_dir, cur_dir, f) for f in [
        'ixgemmblaslt_kernel.cu',
        'ixgemmblaslt.cpp',
        ]],
    extra_ldflags=['-lcudart', 
                    '-lcublasLt', 
                ],
    with_cuda = True,
    verbose = True,
    build_directory = build_dir,
    )
print("build ixgemmblaslt ok")


__all__ = ['gemm88']

