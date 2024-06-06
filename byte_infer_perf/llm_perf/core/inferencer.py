from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List

from llm_perf.core.generation import GenerateConfig, GenerateRequest, GenerateResult
from llm_perf.core.generation import ResultQueue
from llm_perf.utils.logger import logger


class PacketStatus(Enum):
    ERROR = -1
    PENDING = 0
    RUNNING = 1
    FINISH = 2


class CoreInferencer(ABC):
    """
    Inference class
    """

    @dataclass
    class Task:
        request: GenerateRequest
        state: PacketStatus
        generate_ids: List[int]

        def __init__(self, request: GenerateRequest):
            self.request = request
            self.result_queue = ResultQueue()
            self.state = PacketStatus.PENDING
            self.generate_ids = []
            self.exception = None
        
        def add_result(self, res: GenerateResult):
            self.generate_ids.append(res.token_id)
            self.result_queue.put(res)

        async def get_result(self) -> GenerateRequest:
            return await self.result_queue.get()
        
        def finish(self) -> None:
            self.state = PacketStatus.FINISH
            self.result_queue.put(None)

        def error(self) -> None:
            self.state == PacketStatus.ERROR

        def is_finished(self) -> bool:
            return self.state == PacketStatus.FINISH

        def return_q_empty(self) -> bool:
            return self.result_queue.empty()
        
    def __init__(self) -> None:
        super().__init__()



    @abstractmethod
    def infer(self, tasks: List["CoreInferencer.Task"]):
        raise NotImplementedError