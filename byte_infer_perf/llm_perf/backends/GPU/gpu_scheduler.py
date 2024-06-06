import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import torch


from llm_perf.core.scheduler import CoreScheduler
from llm_perf.core.inferencer import CoreInferencer
from llm_perf.core.sampler import CoreSampler
from llm_perf.backends.GPU.gpu_inferencer import GpuInferencer
from llm_perf.utils.logger import logger


class GpuScheduler(CoreScheduler):
    def __init__(
        self,
        inferencer: CoreInferencer,
        sampler: CoreSampler,
        xpu_cfg
    ) -> None:
        super().__init__(inferencer, sampler)

        self.max_batch_size = xpu_cfg["max_batch_size"]


    @torch.inference_mode()
    def scheduler_loop(self):
        batch: List[CoreInferencer.Task] = []
        while self.started:
            # 1. select batch --> batch
            batch = self.select_batch(batch)
            if not batch:
                with self.task_queue.not_empty:
                    self.task_queue.not_empty.wait(0.1)
                continue

            logger.debug(f"get batch size: {len(batch)}")

            # 2. do inference -> logits
            outputs = self.inferencer.infer(batch)

            # 3. sample logits -> tokens
            next_tokens, softmax_out = self.sampler.sample(
                tasks=batch, logits=outputs["last_logits"]
            )

            # 4.postprocess -> gen result
            generation_results = self.sampler.postprocess(
                tasks=batch,
                infer_outputs=outputs,
                next_tokens=next_tokens,
            )

            # 5. add result to task
            for i, gen_res in enumerate(generation_results):
                batch[i].add_result(gen_res)
                if gen_res.finish_reason:
                    batch[i].finish()

            # 6. is not finished -> remain
            remained: List[CoreInferencer.Packet] = []
            for task in batch:
                if not task.is_finished():
                    remained.append(task)
            batch = remained
    
    def select_batch(self, 
        batch: CoreInferencer.Task
    ):
        batching_size: int = len(batch)
        new_select_packets: List[CoreInferencer.Task] = []

        while not self.task_queue.empty():
            if batching_size == self.max_batch_size:
                break
            batching_size += 1
            new_select_packets.append(self.task_queue.get())

        return batch + new_select_packets
