import os
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from llm_perf.backends.GPU.gpu_engine import GpuEngine
from llm_perf.backends.GPU.gpu_sampler import GpuSampler
from llm_perf.core.scheduler import CoreScheduler
from llm_perf.backends.GPU.gpu_scheduler import GpuScheduler


def setup_scheduler(
    model_cls, 
    model_config: Dict[str, Any], 
    tokenizer: PreTrainedTokenizer, 
    max_batch_size: int, 
    **kwargs
) -> CoreScheduler:
    
    # init distributed env if needed
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl", world_size=world_size, rank=local_rank
        )

    # create engine
    engine = GpuEngine(model_cls, model_config, tokenizer)

    # create sampler
    sampler = GpuSampler()

    # create scheduler
    scheduler = GpuScheduler(
        engine=engine, 
        sampler=sampler, 
        max_batch_size=max_batch_size
    )

    return scheduler