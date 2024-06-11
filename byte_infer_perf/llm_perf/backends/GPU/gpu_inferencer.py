import os
from typing import Dict, List, Any
from dataclasses import dataclass

from llm_perf.core.generation import GenerateRequest
from llm_perf.core.inferencer import CoreInferencer
from llm_perf.backends.GPU.gpu_mp_engine import GpuMpEngine
from llm_perf.utils.logger import logger

class GpuInferencer(CoreInferencer):
    def __init__(self, model_impl, xpu_cfg):
        super().__init__()

        self.tp_size = xpu_cfg["tp_size"]
        self.pad_token_id = xpu_cfg["pad_token_id"]
        self.mp_engine = GpuMpEngine(self.tp_size, model_impl, xpu_cfg)

    def prepare_inputs(self, tasks: List["CoreInferencer.Task"]):
        all_input_ids = []
        all_position_ids = []
        all_attention_mask = []

        max_seq_len = -1
        for task in tasks:
            cur_id_len = len(task.request.input_ids) + len(task.generate_ids)
            max_seq_len = cur_id_len if cur_id_len > max_seq_len else max_seq_len

        for task in tasks:
            cur_id_len = len(task.request.input_ids) + len(task.generate_ids)
            pad_len = max_seq_len - cur_id_len
            # using left padding
            input_ids = (
                [self.pad_token_id] * pad_len + 
                task.request.input_ids + 
                task.generate_ids
            )
            pos_ids = (
                [i for i in range(max_seq_len)]
            )
            attention_mask = (
                [0] * pad_len + 
                [1] * cur_id_len
            )
            all_input_ids.append(input_ids)
            all_position_ids.append(pos_ids)
            all_attention_mask.append(attention_mask)
        
        # create model_inputs
        model_inputs = {}
        model_inputs["input_ids"] = all_input_ids
        model_inputs["position_ids"] = all_position_ids
        model_inputs["attention_mask"] = all_attention_mask

        return model_inputs


    def infer(self, tasks: List["CoreInferencer.Task"]):
        input_dict = self.prepare_inputs(tasks)
        outputs = self.mp_engine.mp_forward(input_dict)

        input_logits = outputs.logits[..., :-1, :].contiguous()
        next_tokens_logits = outputs.logits[:, -1, :].contiguous()
        logger.debug(
            f"tensor shape: {outputs.logits.shape}\n"
            f"next tokens logits: {next_tokens_logits.shape}\n"
            f"input logits: {input_logits.shape}\n"
        )
        return {
            "input_logits": input_logits,
            "last_logits": next_tokens_logits,
        }