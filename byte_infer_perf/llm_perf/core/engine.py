from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List

import torch

from llm_perf.core.common import Packet


class CoreEngine(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def init_inference(self, model: torch.nn.Module):
        """Initialize inference engine, load model and do compile if needed."""
        raise NotImplementedError

    @abstractmethod
    def do_inference(self, packets: List[Packet]):
        """Real inference function, do inference and return logits

        Args:
            packets: batch packets of inference

        Return:
            inference results of batch packets
        """
        raise NotImplementedError
