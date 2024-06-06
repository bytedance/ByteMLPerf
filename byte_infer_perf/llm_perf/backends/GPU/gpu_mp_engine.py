import os
from multiprocessing import Queue

import torch
import torch.nn as nn

from llm_perf.core.mp_engine import CoreMpEngine
from llm_perf.utils.logger import logger

class GpuMpEngine(CoreMpEngine):
    def __init__(self, world_size: int, model_impl: nn.Module, xpu_cfg) -> None:
        super().__init__(world_size, model_impl, xpu_cfg)


    def build_inputs(self, forward_inputs):
        forward_inputs["input_ids"] = torch.tensor(forward_inputs["input_ids"]).cuda()
        forward_inputs["position_ids"] = torch.tensor(forward_inputs["position_ids"]).cuda()
        forward_inputs["attention_mask"] = torch.tensor(forward_inputs["attention_mask"]).cuda()
        return forward_inputs
        
    @torch.no_grad()
    def mp_loop_worker(
        self, 
        local_rank: int, 
        world_size: int, 
        input_queue: Queue, 
        output_queue: Queue, 
        model_impl, 
        xpu_config
    ):
        try:
            torch.manual_seed(1)

            # set rank and world_size
            os.environ["RANK"] = str(local_rank)
            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
            

            # set device
            torch.cuda.set_device(local_rank)

            # create and init model based on model_impl and xpu_config
            model = model_impl(xpu_config)
        
            # current rank is ready
            output_queue.put("ready")
            logger.info(f"{local_rank}/{world_size} rank is ready")

            # model process loop
            while True:
                (
                    forward_inputs,
                ) = input_queue.get(block=True)

                # model forward
                inputs = self.build_inputs(forward_inputs)
                logits = model.forward(inputs)

                if local_rank == 0:
                    output_queue.put(logits)
                torch.cuda.synchronize()

        except Exception as e:
            logger.exception(f"[BUG] engine _load_and_listen failed, no more requests will be handled. {e}")
            output_queue.put(RuntimeError("[BUG] fatal exception in model subprocess"))
            

    def mp_forward(self, *args):
        for i in range(self.world_size):
            self._input_queues.put(args, True)
        return self._output_queues.get(True)