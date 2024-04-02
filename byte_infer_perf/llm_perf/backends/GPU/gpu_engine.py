from typing import Any, Dict, List

import torch
from torch import distributed as dist
from transformers import PreTrainedTokenizer

from llm_perf.backends.GPU.common import GPUMultiProcessMsgr
from llm_perf.core.common import Packet
from llm_perf.core.engine import CoreEngine
from llm_perf.utils.logger import logger



class GpuEngine(CoreEngine):
    def __init__(
        self, 
        model_cls, 
        model_config: Dict[str, Any], 
        tokenizer: PreTrainedTokenizer, 
        **kwarg
    ) -> None:
        super().__init__()
        
        # check dist
        if dist.is_initialized():
            self.local_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.local_rank = 0
            self.world_size = 1

        # init multiprocessr msgr
        if self.world_size > 1:
            self.mlp_manager = GPUMultiProcessMsgr(
                self.local_rank, self.world_size, "MultiProcessMsgr"
            )

        # aux configs
        self.pad_token_id = tokenizer.pad_token_id
        self.model_cls = model_cls
        self.model_config = model_config

        # init inference
        self.init_inference()

        
    def init_inference(self):
        # set device
        torch.cuda.set_device(self.local_rank)

        # create model based on model_name
        model_name = self.model_config['model_name']
        if model_name == 'gpt2':
            pass
        elif model_name == 'chatglm':
            from llm_perf.model_zoo.chatglm import ChatGLMConfig
            self.model = self.model_cls.from_pretrained(
                self.model_config["model_path"], config=ChatGLMConfig(**self.model_config["network"])
            )
        elif model_name == 'chatglm2':
            from llm_perf.model_zoo.chatglm2 import ChatGLMConfig
            self.model = self.model_cls.from_pretrained(
                self.model_config["model_path"], config=ChatGLMConfig(**self.model_config["network"])
            )
        elif model_name == 'llama2':
            from llm_perf.model_zoo.llama2 import LlamaConfig
            self.model = self.model_cls.from_pretrained(
                self.model_config["model_path"], config=LlamaConfig(**self.model_config["network"])
            )
        else:
            raise ValueError(f'Unknown model name: {model_name}')

        # set model
        self.model.eval()
        self.model.half().cuda()

        logger.info(f"cuda model {self.model_config['model_path']} loaded {self.model}")



    def broadcast_inputs(self, *args):
        if self.world_size <= 1:
            return args
        
        if self.local_rank == 0:
            self.mlp_manager.broadcast(args)
            return args
        else:
            inputs = self.mlp_manager.receive()
            return inputs


    def prepare_inputs(self, batch: List[Packet]) -> Dict:
        # TODO: prepare inputs based on model type
        all_input_ids = []
        all_position_ids = []

        max_seq_len = -1
        for packet in batch:
            cur_id_len = len(packet.request.input_ids) + len(packet.generate_ids)
            max_seq_len = cur_id_len if cur_id_len > max_seq_len else max_seq_len

        for packet in batch:
            cur_id_len = len(packet.request.input_ids) + len(packet.generate_ids)
            pad_len = max_seq_len - cur_id_len
            input_ids = (
                packet.request.input_ids + 
                packet.generate_ids + 
                [self.pad_token_id] * pad_len
            )
            all_input_ids.append(input_ids)
            all_position_ids.append([i for i in range(max_seq_len)])

        model_inputs = {
            "past_key_values": None, 
            "attention_mask": None, 
            "use_cache": None
        }
        model_inputs["input_ids"] = all_input_ids
        model_inputs["position_ids"] = all_position_ids

        model_name = self.model_config['model_name']
        if model_name == 'chatglm2':
            model_inputs["return_last_logit"] = False
        return model_inputs


    def do_inference(self, packets: List[Packet]):
        # set device
        torch.cuda.set_device(self.local_rank)

        # prepare inputs for each process
        model_inputs = self.prepare_inputs(packets) if self.local_rank == 0 else None
        model_inputs = self.broadcast_inputs(model_inputs)[0]

        # convert inputs to torch tensor
        model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"])
        model_inputs["position_ids"] = torch.tensor(model_inputs["position_ids"])

        # cpu to cuda
        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor):
                model_inputs[k] = v.cuda()

        # model forward
        outputs = self.model(**model_inputs)

        # rank 0: return logits
        # other rank: return
        if self.local_rank == 0:
            input_logits = outputs.logits[..., :-1, :].contiguous()
            next_tokens_logits = outputs.logits[:, -1, :].contiguous()

            logger.info(
                f"tensor shape: {outputs.logits.shape}\n"
                f"next tokens logits: {next_tokens_logits.shape}\n"
                f"input logits: {input_logits.shape}\n"
            )

            return {
                "input_logits": input_logits,
                "last_logits": next_tokens_logits,
            }