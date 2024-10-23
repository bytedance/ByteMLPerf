import triton
import triton.language as tl

from triton.triton_utils import compile_and_cache_kernels
#from blade_llm.module.kernel_backend import get_autotune_triton_kernel


@triton.jit
def _abs_max(val1, val2):
    val1_abs = tl.abs(val1)
    val2_abs = tl.abs(val2)
    if val1_abs >= val2_abs:
        return val1_abs
    else:
        return val2_abs


_dynamic_quant_configs = [
    triton.Config(
        {},
        num_warps=warps,
    )
    for warps in [2, 4, 8, 16]
]


@triton.jit
def _triton_dynamic_quantize_kernel(
    output_ptr,
    input_ptr,
    scale_ptr,
    stride_outputm,
    stride_outputn,
    stride_inputm,
    stride_inputn,
    n_elements,
    N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, N)
    mask = offsets < n_elements
    input_ptrs = input_ptr + pid * stride_inputm + offsets
    input_vals = tl.load(input_ptrs, mask=mask, other=1e-6)
    abs_max_f = tl.reduce(input_vals, 0, _abs_max)
    dynamic_per_token_scale = 127.0 / abs_max_f
    precison_mask = tl.where(input_vals > 0, 0.5, -0.5)
    output_vals = (input_vals * dynamic_per_token_scale + precison_mask).to(tl.int8)
    output_ptrs = output_ptr + pid * stride_outputm + offsets
    tl.store(output_ptrs, output_vals, mask=mask)
    tl.store(scale_ptr + pid, abs_max_f / 127.0)


def triton_dynamic_quantize(out, input, scale):
    assert input.is_contiguous(), "input must be contiguous"
    num_tokens = input.size(0)
    hidden_size = input.size(1)
    # tl.reduce requires the number of elements
    # must be power-of-two
    hidden_size_padded = triton.next_power_of_2(int(hidden_size))
    kwargs = [
        out,
        input,
        scale,
        out.stride(0),
        out.stride(1),
        input.stride(0),
        input.stride(1),
        input.size(1),
    ]
    grid = (num_tokens, 1, 1)
    const_kwargs = {"N": hidden_size_padded}
    method_name = "dynamic_quant_" + str(hidden_size_padded)
    if get_autotune_triton_kernel():
        dynamic_quant = triton.autotune(configs=_dynamic_quant_configs, key=['N'])(_triton_dynamic_quantize_kernel)
    else:
        dynamic_quant = _triton_dynamic_quantize_kernel
    compile_and_cache_kernels(dynamic_quant, method_name, grid, kwargs, const_kwargs)

