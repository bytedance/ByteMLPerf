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
reduction_ops
"""
class LayerNormOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
            
            self.input_tensor_info = {
                "src": OpTensorInfo(
                    shape=[self.batch_size, self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ), 
                "weight": OpTensorInfo(
                    shape=[self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
                "bias": OpTensorInfo(
                    shape=[self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
            }
            self.output_tensor_info = {
                "dst": OpTensorInfo(
                    shape=[self.batch_size, self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
            }
        else:
            raise NotImplementedError


        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self._run_func = self.layer_norm_run
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )

    def layer_norm_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        weight = tensor_mapping["weight"]
        bias = tensor_mapping["bias"]
        dst = torch.nn.functional.layer_norm(src, (self.dim_size,), weight=weight, bias=bias)
        return dst




class ReduceMaxOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]

            self.input_tensor_info = {
                "src": OpTensorInfo(
                    shape=[self.batch_size, self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
            }
            self.output_tensor_info = {
                "max_value": OpTensorInfo(
                    shape=[self.batch_size, 1],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
                "max_indices": OpTensorInfo(
                    shape=[self.batch_size, 1],
                    dtype=torch.int32,
                    device=self.backend.get_device(),
                )
            }
        else:
            raise NotImplementedError

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size
        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self._run_func = self.reduce_max_run    
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )

    def reduce_max_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        max_value, max_indices = torch.max(src, dim=-1, keepdim=True)
        return max_value, max_indices





class ReduceMinOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
            self.input_tensor_info = {
                "src": OpTensorInfo(
                    shape=[self.batch_size, self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
            }
            self.output_tensor_info = {
                "min_value": OpTensorInfo(
                    shape=[self.batch_size, 1],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
                "min_indices": OpTensorInfo(
                    shape=[self.batch_size, 1],
                    dtype=torch.int32,
                    device=self.backend.get_device(),
                )
            }
        else:
            raise NotImplementedError

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self._run_func = self.reduce_min_run
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )

    def reduce_min_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        min_value, min_indices = torch.min(src, dim=-1, keepdim=True)
        return min_value, min_indices



class ReduceSumOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)
        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
            self.input_tensor_info = {
                "src": OpTensorInfo(
                    shape=[self.batch_size, self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
            }
            self.output_tensor_info = {
                "sum": OpTensorInfo(
                    shape=[self.batch_size, 1],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                )
            }
        else:
            raise NotImplementedError
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size
        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes
        self._run_func = self.reduce_sum_run
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )
    def reduce_sum_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        sum = torch.sum(src, dim=-1, keepdim=True)
        return sum

    



class SoftmaxOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]

        if self.arg_type == "default":
            self.dtype = self.args_dict["dtype"]
            self.torch_dtype = getattr(torch, self.dtype)

            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
            self.dim = -1

            self.input_tensor_info = {
                "src": OpTensorInfo(
                    shape=[self.batch_size, self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                )
            }
            self.output_tensor_info = {
                "dst": OpTensorInfo(
                    shape=[self.batch_size, self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                )
            }
        elif self.arg_type == "llm":
            self.dtype = self.args_dict["dtype"]
            self.torch_dtype = getattr(torch, self.dtype)

            self.batch_size = self.args_dict["batch_size"]
            self.head_num = self.args_dict["head_num"]
            self.q_seq_len = self.args_dict["q_seq_len"]
            self.kv_seq_len = self.args_dict["kv_seq_len"]
            self.dim = -1

            self.input_tensor_info = {
                "src": OpTensorInfo(
                    shape=[self.batch_size, self.head_num, self.q_seq_len, self.kv_seq_len],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                )
            }
            self.output_tensor_info = {
                "dst": OpTensorInfo(
                    shape=[self.batch_size, self.head_num, self.q_seq_len, self.kv_seq_len],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                )
            }
        else:
            raise NotImplementedError
        
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = self.input_tensor_size
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.input_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self._run_func = self.softmax_run
        self._create_in_out_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )

    def softmax_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = torch.nn.functional.softmax(src, dim=self.dim)
        return dst



class TopkOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
            self.k = self.args_dict["k"]

            self.input_tensor_info = {
                "src": OpTensorInfo(
                    shape=[self.batch_size, self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                )
            }
            self.output_tensor_info = {
                "value": OpTensorInfo(
                    shape=[self.batch_size, self.k],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ), 
                "indice": OpTensorInfo(
                    shape=[self.batch_size, self.k],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                )
            }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = self.input_tensor_size
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.input_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self._run_func = self.topk_run
        self._create_in_out_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )
        
    def topk_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        value, indice = torch.topk(src, self.k, dim=-1, largest=True, sorted=False)
        return value, indice
    


