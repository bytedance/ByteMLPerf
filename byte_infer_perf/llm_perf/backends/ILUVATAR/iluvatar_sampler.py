from typing import Any, Dict, List, Tuple, Union

import torch

from llm_perf.core.generation import GenerateResult
from llm_perf.core.engine import CoreEngine
from llm_perf.core.sampler import CoreSampler

from vllm import SamplingParams


class IluvatarSampler(CoreSampler):
    def __init__(self) -> None:
        super().__init__()


    def sample(self, packets: List[CoreEngine.Packet], logits: torch.FloatTensor) -> List[int]:
        raise NotImplementedError


    def postprocess(
        self,
        packets: List[CoreEngine.Packet],
        infer_outputs: Dict[str, torch.FloatTensor],
        next_tokens: List[int],
    ) -> List[GenerateResult]:
        
        raise NotImplementedError
    

    def sampling(self, max_new_tokens):
        return SamplingParams(n=1, temperature=0.8, top_k=5, presence_penalty=0.2, max_tokens=max_new_tokens, ignore_eos=True)
