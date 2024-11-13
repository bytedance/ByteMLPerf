grab form vllm

1. to get gemm shapes to be tuned, replace F.linear by tgemm.mm under rocm_kernels/tuned_gemm.py, 
run
    VLLM_TUNE_GEMM=1 python {workload_tests}
shapes will be captured in rocm_kernels/configs/untuned_gemm.csv

2. to tune gemms in rocm_kernels/configs/untuned_gemm.csv,
run 
    python rocm_kernels/gradlib/gradlib/gemm_tuner.py --tuned_file rocm_kernels/configs/tuned_gemm.csv  --input_file rocm_kernels/configs/untuned_gemm.csv

3. then run your test as normal~