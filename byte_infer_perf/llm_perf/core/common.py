from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
from typing import Any, Dict, List

import torch.multiprocessing as mp


class PacketStatus(Enum):
    ERROR = -1
    PENDING = 0
    RUNNING = 1
    FINISH = 2


@dataclass
class GenerateConfig:
    min_new_tokens: int = 0
    max_new_tokens: int = 0
    top_k: int = 0
    top_p: float = 1.0
    temperature: float = 1.0
    presence_penalty: float = 1.0
    eos_token_id: int = -1
    pad_token_id: int = -1
    get_input_logits: bool = False


@dataclass
class GenerateRequest:
    input_ids: List[int]
    generate_config: GenerateConfig


@dataclass
class GenerateResult:
    token_id: int
    last_logits: List[float] = field(default_factory=list)
    input_logits: List[float] = field(default_factory=list)


class Packet:
    request: GenerateRequest
    # result_queue: mp.Queue
    state: PacketStatus
    generate_ids: List[int]

    def __init__(self, request: GenerateRequest):
        self.request = request
        self.result_queue = mp.Manager().Queue()
        self.state = PacketStatus.PENDING
        self.generate_ids = []

    def add_result(self, res: GenerateResult):
        self.generate_ids.append(res.token_id)
        self.result_queue.put(res)

    def get_result(self) -> GenerateResult:
        result = self.result_queue.get()
        return result

    def finish(self) -> None:
        self.state = PacketStatus.FINISH

    def error(self) -> None:
        self.state = PacketStatus.ERROR

    def is_finished(self) -> bool:
        return self.state == PacketStatus.FINISH

    def result_q_empty(self) -> bool:
        return self.result_queue.empty()
