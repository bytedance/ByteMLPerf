## __all__ is a dict:
##   key is model_name in `model_zoo/chatglm-xx.json`
##   value is vendor specify model impl
# __all__ = {
#     "chatglm" : ChatGLMForConditionalGeneration,
#     "chatglm2" : ChatGLM2ForConditionalGeneration
# }

from typing import Dict, Tuple, Any

import torch
import torch.nn as nn

from .chatglm2 import ChatGLMForConditionalGeneration, ChatGLMConfig

from llm_perf.utils.logger import logger


class GPUChatGLM2(nn.Module):
    def __init__(self, xpu_cfg: Dict[str, Any]) -> None:
        super().__init__()

        model_config = xpu_cfg["model_config"]
        model_name = model_config["model_name"]
        model_path = model_config["model_path"]
        model_network = model_config["network"]

        self.model = ChatGLMForConditionalGeneration.from_pretrained(
            model_path, 
            config=ChatGLMConfig(**model_network)
        )
        self.model.eval()
        self.model.half().cuda()
        logger.info(f"cuda model {model_path} loaded {self.model}")
        
    def forward(self, inputs):
        outputs = self.model(**inputs)
        return outputs


__all__ = {
    "chatglm2": GPUChatGLM2
}