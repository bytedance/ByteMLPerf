from typing import Any, Dict

from llm_perf.core.scheduler import CoreScheduler
from llm_perf.backends.ILUVATAR.iluvatar_engine import IluvatarEngine
from llm_perf.backends.ILUVATAR.iluvatar_sampler import IluvatarSampler
from llm_perf.backends.ILUVATAR.iluvatar_scheduler import IluvatarScheduler
from llm_perf.utils.logger import logger

def setup_scheduler(
    model_config: Dict[str, Any], 
    pad_token_id, max_batch_size, 
    **kwargs
) -> CoreScheduler:
    # create engine
    engine = IluvatarEngine(model_config, pad_token_id)

    # create sampler
    sampler = IluvatarSampler()

    # create scheduler
    scheduler = IluvatarScheduler(
        engine=engine, 
        sampler=sampler, 
        max_batch_size=max_batch_size
    )

    return scheduler
