import queue
from multiprocessing import managers

import torch

from llm_perf.core.engine import MultiProcessMsgr


class IluvatarMultiProcessMsgr(MultiProcessMsgr, managers.BaseManager):
    def __init__(self, local_rank: int, world_size: int, name: str):
        self.rank = local_rank
        self.world_size = world_size

        def make_message_queue(rank):
            if rank != 0:
                return None
            new_queue = queue.Queue()
            return lambda: new_queue

        for i in range(1, world_size):
            self.register(f"message_queue_{i}", callable=make_message_queue(local_rank))
        if local_rank == 0:
            super().__init__(authkey=name.encode("utf-8"))
            self.start()
            addr = [self.address]
            torch.distributed.broadcast_object_list(addr, device=f"cuda:{local_rank}")
            self.msg_queue_list = [
                getattr(self, f"message_queue_{rank}")()
                for rank in range(1, world_size)
            ]
        else:
            addr = [None]
            torch.distributed.broadcast_object_list(addr, device=f"cuda:{local_rank}")
            super().__init__(address=addr[0], authkey=name.encode("utf-8"))
            self.connect()
            self.msg_queue = getattr(self, f"message_queue_{local_rank}")()

    def broadcast(self, obj):
        assert (
            self.rank == 0
        ), f"InterProcessMessager broadcast_message only allow rank0 to call!"
        for rank in range(1, self.world_size):
            self.msg_queue_list[rank - 1].put(obj)

    def receive(self):
        assert (
            self.rank > 0
        ), f"InterProcessMessager receive_message don't allow rank0 to call!"
        return self.msg_queue.get()