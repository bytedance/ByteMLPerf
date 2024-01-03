import os
from typing import Any, Dict, List

from llm_perf.core import common


class Packet(common.Packet):
    def __init__(self, request: common.GenerateRequest):
        common.Packet.__init__(self, request)

        self.generation_start_time = None

    def _is_finished(self) -> bool:
        return self.is_finished()
