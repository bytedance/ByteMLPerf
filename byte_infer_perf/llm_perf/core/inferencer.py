import asyncio
from typing import Any, AsyncIterable, Dict, Iterable

from transformers import AutoTokenizer, PreTrainedTokenizer

from llm_perf.core.common import GenerateConfig, GenerateRequest, GenerateResult
from llm_perf.core.scheduler import CoreScheduler
from llm_perf.utils.logger import logger

import os
import importlib


# get model impl from orig or vendor 
def get_model_impl(
    model_config: Dict[str, Any], 
    hardware_type: str
):
    # for example, "ChatGLMForConditionalGeneration"
    model_inferface = model_config["model_interface"]
    model_name = model_config["model_name"]

    # Get orig model
    spec = importlib.util.spec_from_file_location(
        model_name, f"llm_perf/model_zoo/{model_name.split('-')[0]}.py"
    )
    base_module_impl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_module_impl)

    orig_model = getattr(base_module_impl, model_inferface)

    # Get vendor model
    vendor_model_path = f"llm_perf/backends/{hardware_type}/model_impl"
    if not os.path.exists(vendor_model_path):
        logger.info(
            f"{vendor_model_path} not exists, {model_inferface} model select <ByteMLPerf base model>"
        )
        return orig_model

    vendor_model_impl = importlib.import_module(
        ".", package=vendor_model_path.replace("/", ".")
    )
    if not model_name in vendor_model_impl.__all__.keys():
        logger.info(
            f"can't find {model_name} in {vendor_model_path}/__init__, model select <ByteMLPerf base model>"
        )
        return orig_model

    vendor_model = vendor_model_impl.__all__[model_name]
    logger.info(
        f"find {model_inferface} in {vendor_model_path}, model select <Vendor model>"
    )
    return vendor_model


class CoreInferencer:
    def __init__(
        self, model_config, 
        hardware_type, max_batch_size
    ) -> None:
        super().__init__()

        # load tokenizer
        tokenizer_path = model_config["tokenizer"]["path"]
        self.add_sep_token = model_config["tokenizer"]["add_sep_token"]
        self.tokenizer : PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_path, 
            local_files_only=True,
            trust_remote_code=True
        )
        logger.info(f'load tokenizer: {tokenizer_path}')
   
        # set up model and scheduler
        setup = importlib.import_module(
            ".setup", package=f"llm_perf.backends.{hardware_type}"
        )
        logger.info(f"import setup: {setup}")

        # load model impl: orig or vendor_custom
        model_impl = get_model_impl(model_config, hardware_type)

        self.scheduler = setup.setup_scheduler(
            model_impl, 
            model_config, 
            tokenizer=self.tokenizer, 
            max_batch_size=max_batch_size
        )
        self.scheduler.start()
        self.warmup()


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
                completion_tokens += 1
                outputs = {
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "choice": {},
                }

                if req.generate_config.get_input_logits:
                    result, perplexity, logits_dump = gen_res
                    if result is None:
                        outputs["choice"].update(
                            {
                                "message": "",
                                "perplexity": perplexity,
                                "logits_dump": logits_dump,
                            }
                        )
                    else:
                        outputs["choice"].update(
                            {
                                "message": self.tokenizer.decode(result.token_id),
                                "perplexity": perplexity,
                                "logits_dump": logits_dump,
                            }
                        )
                else:
                    result: GenerateResult = gen_res
                    outputs["choice"].update(
                        {"message": self.tokenizer.decode(result.token_id)}
                    )

                logger.debug(f"steam inference result: {outputs}")
                yield outputs
        except Exception as e:
            logger.error(f"stream inference error: {e}")
            raise e

    def warmup(self):
        prompt = "中国的首都是哪里？"
        generate_config = {
            "min_new_tokens": 1,
            "max_new_tokens": 512,
            "top_k": 1,
            "temperature": 0.2,
            "presence_penalty": 1.0,
        }
        logger.info(f"warmup prompt: {prompt}\nconfig: {generate_config}")

        async def _steram_warmup():
            message = ""
            async for result in self.streaming_inference(prompt, generate_config):
                message += result["choice"]["message"]
            result["choice"]["message"] = message
            return result

        result = asyncio.run(_steram_warmup())
        logger.info(f"warmup response: {result}")
