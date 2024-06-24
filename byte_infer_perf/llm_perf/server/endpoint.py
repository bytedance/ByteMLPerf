import sys
import json
import pathlib
import asyncio
import importlib
from typing import Any, AsyncIterable, Dict, Iterable

from transformers import AutoTokenizer, PreTrainedTokenizer

from llm_perf.core.generation import GenerateConfig, GenerateRequest, GenerateResult
from llm_perf.core.scheduler import CoreScheduler
from llm_perf.utils.logger import logger


class LLMPerfEndpoint:
    def __init__(self, xpu_cfg) -> None:
        super().__init__()

        self.xpu_cfg = xpu_cfg
        
        model_config = xpu_cfg["model_config"]
        hardware_type = xpu_cfg["hardware_type"]
        
        # load tokenizer
        tokenizer_path = model_config["tokenizer"]["path"]
        self.add_sep_token = model_config["tokenizer"]["add_sep_token"]
        self.tokenizer : PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_path, 
            local_files_only=True,
            trust_remote_code=True
        )
        logger.info(f'load tokenizer: {tokenizer_path}')
        logger.info(f'pad_token_id: {self.tokenizer.pad_token_id}')
        logger.info(f'eos_token_id: {self.tokenizer.eos_token_id}')

        xpu_cfg["pad_token_id"] = self.tokenizer.pad_token_id

        # import setup according to hardware_type
        setup = importlib.import_module(
            ".setup", package=f"llm_perf.backends.{hardware_type}"
        )
        logger.info(f"import setup: {setup}")

        # setup scheduler
        self.scheduler : CoreScheduler = setup.setup_scheduler(xpu_cfg)
        self.scheduler.start()

        self.warmup(xpu_cfg["max_batch_size"])

    def __del__(self):
        self.scheduler.stop()

    def warmup(self, max_batch_size):
        prompt = "中国的首都是哪里？"
        generate_config = {
            "min_new_tokens": 1,
            "max_new_tokens": 512,
            "top_k": 1,
            "get_input_logits": 0
        }
        logger.info(f"warmup prompt: {prompt}, config: {generate_config}")

        async def _steram_warmup():
            message = ""
            async for result in self.streaming_inference(prompt, generate_config):
                message += result["choice"]["message"]
            result["choice"]["message"] = message
            return result

        async def _multiple_warmup():
            tasks = []
            for _ in range(max_batch_size):
                tasks.append(_steram_warmup())
            res = await asyncio.gather(*tasks)
            return res

        single_result = asyncio.run(_steram_warmup())
        logger.info(f"single warmup response: {single_result}\n")

        multiple_result = asyncio.run(_multiple_warmup())
        logger.info(f"multiple warmup reponse: {multiple_result}\n")

    async def prepare_request(
        self, prompt: str, generate_config: Dict[str, Any]
    ) -> GenerateRequest:
        input_ids = self.tokenizer.encode(prompt)        
        if self.add_sep_token:
            input_ids.append(self.tokenizer.sep_token_id)

        # create generate config
        config = GenerateConfig(
            min_new_tokens=generate_config.get("min_new_tokens", 0),
            max_new_tokens=generate_config.get("max_new_tokens", 0),
            top_k=generate_config.get("top_k", 0),
            top_p=generate_config.get("top_p", 1.0),
            temperature=generate_config.get("temperature", 1.0),
            presence_penalty=generate_config.get("presence_penalty", 1.0),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            get_input_logits=bool(generate_config.get("get_input_logits", 0)),
        )
        req = GenerateRequest(input_ids, config)
        return req

    async def streaming_inference(
        self, 
        prompt: str, 
        generate_config: Dict[str, Any]
    ) -> AsyncIterable[Dict[str, Any]]:
        try:
            # create GenerateRequest object
            req = await self.prepare_request(prompt, generate_config)

            prompt_tokens = len(req.input_ids)
            completion_tokens = 0

            async for gen_res in self.scheduler.generate(req):
                result = gen_res["result"]
                if result is not None:
                    completion_tokens += 1

                infer_outputs = {
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "choice": {
                        "message": ""
                    }
                }

                if result is not None:
                    infer_outputs["choice"].update(
                        {
                            "message": self.tokenizer.decode(result.token_id), 
                            "wait_time": result.wait_time, 
                            "model_time": result.model_time, 
                            "post_process_time": result.post_process_time
                        }
                    )

                if req.generate_config.get_input_logits:
                    infer_outputs["choice"].update(
                        {
                            "perplexity": gen_res["perplexity"], 
                            "logits_dump": gen_res["dump_file"]
                        }
                    )

                logger.debug(f"steam inference result: {infer_outputs}")
                yield infer_outputs
        except Exception as e:
            logger.error(f"stream inference error: {e}")
            raise e


