import sys
import pathlib
import torch
from functools import partial

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent.parent
sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp


"""
world_size: int

default:    [batch_size, dim_size]
llm:        [batch_size, q_seq_len, hidden_size]
        --> [num_tokens, hidden_size]
"""

class AllReduceOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default", "llm"]:
            raise NotImplementedError
        
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.world_size = self.args_dict["world_size"]

        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
        elif self.arg_type == "llm":
            self.batch_size = self.args_dict["num_tokens"]
            self.dim_size = self.args_dict["hidden_size"]

        # [batch_size * dim_size]
        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }

        # inplace operation
        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = 0
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.input_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes


        self.algo_size = self.input_tensor_size
        self.bus_size = 2 * (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = self.batch_size * self.dim_size

        
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )

        self._run_func = self.all_reduce_run

    def all_reduce_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dist_module = self.backend.get_dist_module()
        dist_module.all_reduce(src, op=dist_module.ReduceOp.SUM, group=self.op_group)
        return src


