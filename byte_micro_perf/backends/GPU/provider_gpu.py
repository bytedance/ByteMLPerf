import importlib
import traceback

GPU_PROVIDER = {}


# https://github.com/Dao-AILab/flash-attention
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    GPU_PROVIDER["support_fa2"] = {
        "fa2": importlib.metadata.version("flash_attn")
    }
except:
    pass


# https://github.com/Dao-AILab/flash-attention
try:
    from flash_attn_interface import flash_attn_func
    GPU_PROVIDER["support_fa3"] = {
        "fa3": importlib.metadata.version("flash_attn_3"),
    }
except:
    pass


# https://github.com/vllm-project/vllm
try:
    import vllm
    GPU_PROVIDER["support_vllm"] = {
        "vllm": importlib.metadata.version("vllm"),
    }
except:
    pass


# https://github.com/flashinfer-ai/flashinfer
try:
    import flashinfer
    GPU_PROVIDER["support_flashinfer"] = {
        "flashinfer": importlib.metadata.version("flashinfer-python"),
    }
except:
    print(traceback.print_exc)

