import sys
import pathlib
import torch


FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger
from core.utils import OpTensorInfo, OpSizeInfo, calc_tensor_size
from core.op import BasicOp



"""
reduction_ops
"""

class SoftmaxOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)
        self.args_type = self.args_dict["args_type"]

        if self.args_type == "llm":
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
            self.output_tensor_info = {}
        else:
            raise NotImplementedError


        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = self.input_tensor_size
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.input_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self._run_func = self.softmax_run

    def softmax_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = torch.nn.functional.softmax(src, dim=self.dim)
        return dst




