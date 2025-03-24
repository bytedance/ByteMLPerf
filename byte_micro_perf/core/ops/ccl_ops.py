import sys
import pathlib
import torch
from functools import partial

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger
from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp



"""
ccl ops
"""
class AllGatherOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)
        
        self.world_size = self.args_dict["world_size"]
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size // self.world_size],
                dtype=self.torch_dtype, 
                device=self.backend.get_device(),
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype, 
                device=self.backend.get_device(),
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.output_tensor_size
        self.bus_size = (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = 0

        self._run_func = self.all_gather_run

    def all_gather_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dist_module = self.backend.get_dist_module()
        dist_module.all_gather_into_tensor(dst, src, group=self.op_group)
        return dst

    def is_concurrent():
        return True



class AllReduceOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        self.world_size = self.args_dict["world_size"]
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_device(),
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_device(),
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = 0
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.input_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.input_tensor_size
        self.bus_size = 2 * (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = 0

        self._run_func = self.all_reduce_run
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )


    def all_reduce_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dist_module = self.backend.get_dist_module()
        dist_module.all_reduce(src, op=dist_module.ReduceOp.SUM, group=self.op_group)
        return src

    def is_concurrent():
        return True


class AlltoAllOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        self.world_size = self.args_dict["world_size"]
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_device(),
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_device(),
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.output_tensor_size
        self.bus_size = (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = 0

        self._run_func = self.all_to_all_run

    def all_to_all_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dist_module = self.backend.get_dist_module()
        dist_module.all_to_all_single(dst, src, group=self.op_group)
        return dst

    def is_concurrent():
        return True
        



class BroadcastOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        self.world_size = self.args_dict["world_size"]
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.world_size, self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_device(),
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.world_size, self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_device(),
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = 0
        self.tensor_size = self.input_tensor_size + self.output_tensor_size
        
        self.read_bytes = self.output_tensor_size / self.world_size
        self.write_bytes = self.output_tensor_size / self.world_size * (self.world_size - 1)
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.tensor_size
        self.bus_size = self.algo_size
        self.calc_flops = 0

        self._concurrent = True
        self._run_func = self.broadcast
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )


    def broadcast(self, tensor_mapping):
        src = tensor_mapping["src"]
        dist_module = self.backend.get_dist_module()
        for i in range(self.world_size):
            dist_module.broadcast(src[i], src=i, group=self.op_group)
        return src

    def is_concurrent():
        return True



class ReduceScatterOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        self.world_size = self.args_dict["world_size"]
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_device(),
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size // self.world_size],
                dtype=self.torch_dtype,
                device=self.backend.get_device(),
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size
        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.input_tensor_size
        self.bus_size = (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = 0

        self._run_func = self.reduce_scatter_run

    def reduce_scatter_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dist_module = self.backend.get_dist_module()
        dist_module.reduce_scatter_tensor(dst, src, op=dist_module.ReduceOp.SUM, group=self.op_group)
        return dst

    def is_concurrent():
        return True



class P2POp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        self.world_size = self.args_dict["world_size"]
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        dist_module = self.backend.get_dist_module()
        self.world_size = self.group_size
        self.local_rank = dist_module.get_rank(group=self.op_group)

        self.next_device = (self.local_rank + 1) % self.world_size
        self.last_device = (self.local_rank - 1 + self.world_size) % self.world_size

        self.input_tensor_info = {
            "send": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_device(),
            )
        }
        self.output_tensor_info = {
            "recv": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_device(),
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.input_tensor_size
        self.bus_size = self.algo_size

        self.calc_flops = 0

        self._run_func = self.p2p_run

    def p2p_run(self, tensor_mapping):
        send_tensor = tensor_mapping["send"]
        recv_tensor = tensor_mapping["recv"]
        dist_module = self.backend.get_dist_module()

        # 0 --> 1
        # 0 --> 1 --> 2 --> 3
        # 0 --> 1 --> 2 --> 3 --> 4 --> 5 --> 6 --> 7 --> 8
        reqs = []
        if self.local_rank != self.world_size - 1:
            reqs.append(dist_module.isend(send_tensor, self.next_device, group=self.op_group))
        if self.local_rank != 0:
            reqs.append(dist_module.irecv(recv_tensor, self.last_device, group=self.op_group))
        for req in reqs:
            req.wait()

    def is_concurrent():
        return True
