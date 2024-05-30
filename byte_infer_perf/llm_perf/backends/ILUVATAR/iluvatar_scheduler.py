import asyncio
from typing import List

import torch

from llm_perf.core.engine import CoreEngine
from llm_perf.core.sampler import CoreSampler
from llm_perf.core.scheduler import CoreScheduler
from llm_perf.backends.ILUVATAR.iluvatar_engine import IluvatarEngine
from llm_perf.backends.ILUVATAR.iluvatar_sampler import IluvatarSampler
from llm_perf.utils.logger import logger
from llm_perf.core.generation import GenerateResult


class IluvatarScheduler(CoreScheduler):
    def __init__(
        self,
        engine: CoreEngine,
        sampler: CoreSampler,
        **kwargs,
    ) -> None:
        super().__init__( 
            engine=engine, 
            sampler=sampler, 
            packet_cls=IluvatarEngine.Packet, 
            **kwargs
        )
        self.max_batch_size = kwargs.get("max_batch_size")

    @torch.inference_mode()
    def scheduler_loop(self):
        batch: List[CoreEngine.Packet] = [] 
        while True:
            # 1. select batch --> batch
            batch = self.select_batch(batch)
            if not batch:
                with self.packet_queue.not_empty:
                    self.packet_queue.not_empty.wait(0.1)
                continue

            logger.debug(f"get batch size: {len(batch)}") 

            # 2. AsyncLLMEngine
            for b in batch:
                max_new_tokens = b.request.generate_config.max_new_tokens
                break

            sampling = self.sampler.sampling(max_new_tokens)
            asyncio.run(self.inference(batch,sampling))

            # 3. is not finished -> remain
            remained: List[CoreEngine.Packet] = []
            for packet in batch:
                if not packet.is_finished():
                    remained.append(packet)

            batch = remained


    def select_batch(self, batch):
        batching_size: int = len(batch)
        new_select_packets: List[CoreEngine.Packet] = []

        while not self.packet_queue.empty():
            if batching_size == self.max_batch_size:
                break

            batching_size += 1
            new_select_packets.append(self.packet_queue.get())

        return batch + new_select_packets
    

    async def inference(self, batch, sampler):
        async for results in self.engine.do_inference(batch, sampler):    
            for j, result in enumerate(results):
                token = result["token_ids"][-1]
                finish_reason = result["finish_reason"]

                if finish_reason == None :
                    gen_res = GenerateResult(token,"")

                else :
                    gen_res = GenerateResult(token,"max_length")

                batch[j].add_result(gen_res)

                if gen_res.finish_reason:
                    batch[j].finish()  
