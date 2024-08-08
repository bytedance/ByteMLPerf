import os
import time
from multiprocessing import Queue
from typing import List

import torch
import torch.nn as nn
import torch.distributed as dist

from llm_perf.core.mp_engine import CoreMpEngine
from llm_perf.utils.logger import logger



# context: 
#   input_ids: [1, s_q]
#   attention_mask = [1, s_q]
#   full_attention_mask = [1, 1, s_q, s_kv] (sq == s_kv)
def get_context_masks(
    input_ids : torch.Tensor, 
    padding_mask : torch.Tensor
):
    # input_ids: [1, q_len]
    # padding_mask = [1, q_len]
    _, q_len = input_ids.shape

    # [1, q_len, q_len]
    full_attention_mask = torch.ones(
        1, q_len, q_len, 
        device=input_ids.device
    )
    full_attention_mask.tril_()
    full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
    full_attention_mask -= padding_mask.unsqueeze(-1) - 1
    full_attention_mask = (full_attention_mask < 0.5).bool()
    full_attention_mask.unsqueeze_(1)
    return full_attention_mask


# decode
#   input_ids: [bs, 1]
#   attention_mask = [bs, 1]
#   full_attention_mask = [bs, 1, 1, s_kv]
def get_decode_masks(
    input_ids : torch.Tensor, 
    all_kv_len: List[int]
):
    # input_ids: [batch_size, 1]
    # padding_mask: [batch_size, 1 + max_kv_len]
    batch_size, q_len = input_ids.shape
    max_qkv_len = q_len + max(all_kv_len)
    
    # [batch_size, 1, max_qkv_len]
    padding_mask = []
    for i in range(batch_size):
        cur_qkv_len = q_len + all_kv_len[i]
        mask_per_batch = [1] * cur_qkv_len + [0] * (max_qkv_len - cur_qkv_len)
        padding_mask.append(mask_per_batch)
    full_attention_mask = torch.tensor(
        padding_mask, 
        device=input_ids.device
    ).unsqueeze_(1)
    full_attention_mask = (full_attention_mask < 0.5).bool()
    full_attention_mask.unsqueeze_(1)
    return full_attention_mask


class GpuMpEngine(CoreMpEngine):
    def __init__(self, world_size: int, model_impl: nn.Module, xpu_cfg) -> None:
        super().__init__(world_size, model_impl, xpu_cfg)


    def build_inputs(self, forward_inputs):
        # list --> torch.Tensor --> cuda
        forward_inputs["input_ids"] = torch.tensor(
            forward_inputs["input_ids"]
        ).cuda()
        forward_inputs["position_ids"] = torch.tensor(
            forward_inputs["position_ids"]
        ).cuda()
        forward_inputs["attention_mask"] = torch.tensor(
            forward_inputs["attention_mask"]
        ).cuda()
        
        is_context = forward_inputs["is_context"]
        if is_context:
            forward_inputs["full_attention_mask"] = get_context_masks(
                forward_inputs["input_ids"],
                forward_inputs["attention_mask"]
            )
        else:
            forward_inputs["full_attention_mask"] = get_decode_masks(
                forward_inputs["input_ids"],
                forward_inputs["all_kv_len"]
            )
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

            # create and init model based on model_impl and xpu_config
            model = model_impl(xpu_config)
            model.init_inference()
        
            # current rank is ready
            output_queue.put("ready")
            logger.info(f"{local_rank}/{world_size} rank is ready")

            # model process loop
            while True:
                (
                    forward_inputs,
                ) = input_queue.get(block=True)
                inputs_dict = self.build_inputs(forward_inputs)
                output_dict = model.forward(inputs_dict)
                torch.cuda.synchronize()
                if local_rank == 0:
                    output_queue.put(output_dict)

        except Exception as e:
            logger.exception(f"[BUG] engine _load_and_listen failed, no more requests will be handled. {e}")
            output_queue.put(RuntimeError("[BUG] fatal exception in model subprocess"))
            

    def mp_forward(self, *args):
        for i in range(self.world_size):
            self._input_queues.put(args, True)
        return self._output_queues.get(True)