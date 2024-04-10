from typing import Any, Dict

from llm_perf.core.scheduler import CoreScheduler
from llm_perf.backends.GPU.gpu_engine import GpuEngine
from llm_perf.backends.GPU.gpu_sampler import GpuSampler
from llm_perf.backends.GPU.gpu_scheduler import GpuScheduler
from llm_perf.utils.logger import logger

def setup_scheduler(
    model_config: Dict[str, Any], 
    pad_token_id, max_batch_size, 
    **kwargs
) -> CoreScheduler:
    # create engine
    engine = GpuEngine(model_config, pad_token_id)

    # create sampler
    sampler = GpuSampler()

    # create scheduler
    scheduler = GpuScheduler(
        engine=engine, 
        sampler=sampler, 
        max_batch_size=max_batch_size
    )

    return scheduler