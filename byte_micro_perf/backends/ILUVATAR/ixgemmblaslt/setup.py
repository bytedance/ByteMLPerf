
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ixgemmblaslt',
    version="1.0.0",
    ext_modules=[
        CUDAExtension('ixgemmblaslt', [
            'ixgemmblaslt.cpp',
            'ixgemmblaslt_kernel.cu',
        ],
        include_dirs=[],
        #define_macros=[('CAL_TFLOPS_TEST', 1)],
        extra_compile_args={
            'cxx': ['-std=c++17'],
            'clang++': ['-std=c++17', ],
        },
        libraries=['cudart', 'cublasLt']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

