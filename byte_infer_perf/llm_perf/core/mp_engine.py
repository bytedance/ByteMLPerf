import os
import atexit
from abc import ABC, abstractmethod

import torch.nn as nn
import torch.multiprocessing as mp

from llm_perf.utils.logger import logger

class CoreMpEngine(ABC):
    def __init__(
        self, 
        world_size: int, 
        model_impl: nn.Module, 
        xpu_cfg
    ) -> None:
        self.world_size = world_size
        self.model_impl = model_impl
        self.xpu_cfg = xpu_cfg

        # https://github.com/pytorch/pytorch/issues/32322
        # https://stackoverflow.com/questions/61939952/mp-set-start-methodspawn-triggered-an-error-saying-the-context-is-already-be
        try:
            mp.set_start_method("spawn", force=True)
        except Exception as e:
            logger.exception(f"failed to set spawn context: {e}")

        self._input_queues = mp.Queue(maxsize=self.world_size)
        self._output_queues = mp.Queue(maxsize=1)

        if os.getenv("MASTER_PORT", "") == "":
            os.environ["MASTER_PORT"] = str(self.find_free_port())

        if os.getenv("MASTER_ADDR", "") == "":
            os.environ["MASTER_ADDR"] = "localhost"

        self._subprocesses = mp.spawn(
            self.mp_loop_worker,
            args=(
                world_size,
                self._input_queues,
                self._output_queues,
                model_impl,
                xpu_cfg,
            ),
            nprocs=world_size,
            join=False,
            daemon=False,
        )

        logger.info("waiting for ranks to be ready")
        for i in range(world_size):
            assert "ready" == self._output_queues.get(block=True)

        logger.info("all ranks are ready and listening, init done")

        atexit.register(self.kill)

    def __del__(self):
        self.kill()

    def kill(self):
        try:
            for p in self._subprocesses.processes:
                if p.is_alive():
                    p.terminate()
        except Exception as e:
            logger.exception(f"{e}, failed to terminate torch mp, which may cause mem leak; ignored...")

    def find_free_port(self):
        import socket
        from contextlib import closing

        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]


    @staticmethod
    @abstractmethod
    def mp_loop_worker(
        local_rank: int, 
        world_size: int, 
        input_queue: mp.Queue, 
        output_queue: mp.Queue, 
        model_impl, 
        xpu_config
    ):
        raise NotImplementedError
    
    @abstractmethod
    def mp_forward(self, *args):
        raise NotImplementedError