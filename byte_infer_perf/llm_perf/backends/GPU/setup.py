import os
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from llm_perf.backends.GPU.gpu_engine_chatglm import GpuEngineChatGLM
from llm_perf.backends.GPU.gpu_engine_chatglm2 import GpuEngineChatGLM2
from llm_perf.backends.GPU.gpu_engine_chinese_llama2 import GpuEngineChineseLlama2
from llm_perf.backends.GPU.gpu_sampler import GpuSampler
from llm_perf.backends.GPU.gpu_scheduler import GpuScheduler
from llm_perf.core.scheduler import CoreScheduler


def setup_scheduler(
    modelcls, model_config: Dict[str, Any], max_batch_size: int, **kwargs
) -> CoreScheduler:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl", world_size=world_size, rank=local_rank
        )

    model_name = model_config["model_name"]
    tokenizer_path = model_config["tokenizer"]["path"]
    add_sep_token = model_config["tokenizer"]["add_sep_token"]

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if model_name == "gpt2":
        pass
    elif model_name == "chatglm":
        engine = GpuEngineChatGLM(modelcls, model_config, tokenizer, max_batch_size)
    elif model_name == "chatglm2":
        engine = GpuEngineChatGLM2(
            modelcls, model_config, tokenizer.pad_token_id, max_batch_size
        )
    elif model_name == "llama2":
        engine = GpuEngineChineseLlama2(
            modelcls, model_config, tokenizer.pad_token_id, max_batch_size
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    sampler = GpuSampler()

    return GpuScheduler(
        engine=engine,
        sampler=sampler,
        tokenizer=tokenizer,
        add_sep_token=add_sep_token,
        max_batch_size=max_batch_size,
    )
