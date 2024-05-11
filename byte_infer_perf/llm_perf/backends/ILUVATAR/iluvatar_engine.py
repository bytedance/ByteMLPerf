import os
import json
import asyncio
from typing import Dict, List

import torch

from llm_perf.core.generation import GenerateRequest
from llm_perf.core.engine import CoreEngine
from llm_perf.backends.ILUVATAR.iluvatar_process_messager import IluvatarMultiProcessMsgr
from llm_perf.utils.logger import logger

from vllm.utils import Counter, random_uuid
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs


class IluvatarEngine(CoreEngine):
    
    class Packet(CoreEngine.Packet):
        def __init__(self, request: GenerateRequest):
            CoreEngine.Packet.__init__(self, request)

            self.generation_start_time = None

        def _is_finished(self) -> bool:
            return self.is_finished()

        @staticmethod
        def prepare_inputs(
            batch: List[CoreEngine.Packet],
            **kwargs
        ) -> Dict:
            model_config = kwargs["model_config"]
            pad_token_id = kwargs["pad_token_id"]

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
                    [pad_token_id] * pad_len
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

            model_name = model_config['model_name']
            if model_name == 'chatglm2':
                model_inputs["return_last_logit"] = False
            return model_inputs


    def __init__(
        self, model_config, pad_token_id, 
        **kwarg
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.pad_token_id = pad_token_id
        self.engine = None
        
        # set up environ
        self.setup()
        
        # init multiprocessr msgr
        if self.world_size > 1:
            self.mlp_manager = IluvatarMultiProcessMsgr(
                self.local_rank, self.world_size, "MultiProcessMsgr"
            )


    def setup(self):
        # init distributed env if needed
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"

        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        
        model = self.model_config['model_path']
        tokenizer = self.model_config["tokenizer"]["path"]
        llm_engine = self.load_model(model, tokenizer)
        self.engine = llm_engine


    def load_model(self, model, tokenizer):
        self.request_counter = Counter()
        
        # Create the AsyncLLMEngine
        engine_args = AsyncEngineArgs(model=model, tokenizer=tokenizer, trust_remote_code=True,)
        llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

        return llm_engine
    

    def broadcast_inputs(self, *args):
        if self.world_size <= 1:
            return args
        
        if self.local_rank == 0:
            self.mlp_manager.broadcast(args)
            return args
        else:
            inputs = self.mlp_manager.receive()
            return inputs


    def prepare_inputs(self, batch: List[CoreEngine.Packet]) -> Dict:
        model_inputs = IluvatarEngine.Packet.prepare_inputs(
            batch, 
            model_config=self.model_config, 
            pad_token_id=self.pad_token_id
        )
        return model_inputs
    

    async def generate(self, samplingparams, request_id, input):  
        async for output in self.engine.generate(None, samplingparams, request_id, input):
            result = output.outputs[0] 
            ret = {"token_ids": result.token_ids, "finish_reason":result.finish_reason}  
            yield json.dumps(ret).encode("utf-8") 


    async def consume_stream(self, samplingparams, input): 
        handler_list = list()
        for i in input:
            request_id = random_uuid()
            handler_list.append(self.generate(samplingparams, str(request_id), i))

        while True:
            data = list()
            try:
                for h in handler_list:
                    result = await anext(h)
                    result.decode('utf-8')
                    result = json.loads(result) 
                    data.append(result)
            except:
                break
            yield data


    async def do_inference(self, packets: List[CoreEngine.Packet], sampler):
        # set device
        torch.cuda.set_device(self.local_rank)

        # prepare inputs for each process
        model_inputs = self.prepare_inputs(packets) if self.local_rank == 0 else None
        model_inputs = self.broadcast_inputs(model_inputs)[0]

        # AsyncLLMEngine
        async for i in self.consume_stream(sampler,model_inputs["input_ids"]):
            yield i
