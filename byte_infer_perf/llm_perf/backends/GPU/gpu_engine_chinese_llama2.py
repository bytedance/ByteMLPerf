import json
import os
import time
from typing import Any, Dict, List

import torch
from torch import distributed as dist
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

from llm_perf.core.common import Packet
from llm_perf.core.engine import CoreEngine
from llm_perf.utils.logger import logger


class GpuEngineChineseLlama2(CoreEngine):
    def __init__(
        self,
        modelcls,
        model_config: Dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        max_batch_size: int,
        **kwarg,
    ) -> None:
        super().__init__()
        if dist.is_initialized():
            self.local_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.local_rank = 0
            self.world_size = 1

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.max_bs = max_batch_size
        self.max_position_embeddings = 2048

        self.init_inference(modelcls, model_config)

    def init_inference(self, modelcls, model_config: Dict[str, Any]):
        torch.cuda.set_device(self.local_rank)
        from llm_perf.model_zoo.llama2 import LlamaConfig

        self.model = modelcls.from_pretrained(
            model_config["model_path"], config=LlamaConfig(**model_config["network"])
        )
        self.model.eval()
        self.model.half().cuda()

        logger.info(f"cuda model {model_config['model_path']} loaded {self.model}")

    def broadcast_inputs(self, **kwargs):
        if self.world_size <= 1:
            return kwargs
        if self.local_rank == 0:
            object_list = [kwargs]
            dist.broadcast_object_list(object_list, device=f"cuda:{self.local_rank}")
        else:
            object_list = [None]
            dist.broadcast_object_list(object_list, device=f"cuda:{self.local_rank}")
        return object_list[0]

    def prepare_inputs(self, batch: List[Packet]) -> Dict:
        all_input_ids = []
        max_seq_len = -1
        for packet in batch:
            if len(packet.request.input_ids) + len(packet.new_token_ids) > max_seq_len:
                max_seq_len = len(packet.request.input_ids) + len(packet.new_token_ids)
        for packet in batch:
            pad_len = max_seq_len - (
                len(packet.request.input_ids) + len(packet.new_token_ids)
            )
            input_ids = (
                packet.request.input_ids
                + packet.new_token_ids
                + [self.pad_token_id] * pad_len
            )
            all_input_ids.append(input_ids)

        input_ids_tensor = torch.tensor(all_input_ids).view(-1, max_seq_len).cuda()
        return self.model.prepare_inputs_for_generation(input_ids_tensor)

    def do_inference(self, packets: List[Packet]):
        torch.cuda.set_device(self.local_rank)
        if self.local_rank == 0:
            model_inputs = self.prepare_inputs(packets)
        else:
            model_inputs = {}

        model_inputs = self.broadcast_inputs(**model_inputs)

        if len(model_inputs) == 0:
            time.sleep(0.1)
            return

        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor):
                model_inputs[k] = v.cuda()

        outputs = self.model(**model_inputs)

        if self.local_rank == 0:
            next_tokens_logits = outputs.logits[:, -1, :].contiguous()
            input_logits = outputs.logits[..., :-1, :].contiguous()

            return {
                "input_logits": input_logits,
                "last_logits": next_tokens_logits,
            }
