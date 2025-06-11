import sys
import pathlib
import torch

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger
from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp


class FlashDecoding(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        # llm args
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type != "llm":
            raise NotImplementedError
        
        # llm phase: decode
        self.phase = self.args_dict.get("phase", "decode")
        if self.phase not in ["decode"]:
            raise NotImplementedError
        
        # dtype: bfloat16
        self.dtype = self.args_dict.get("dtype", "bfloat16")
        if self.dtype!= "bfloat16":
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # parse args
        self.is_causal = self.args_dict.get("is_causal", True)
        if not self.is_causal:
            raise NotImplementedError

        self.q_head_num = self.args_dict.get("q_head_num", 64)
        self.kv_head_num = self.args_dict.get("kv_head_num", 8)
        self.head_dim = self.args_dict.get("head_dim", 128)

        self.batch_size = self.args_dict["batch_size"]
        self.q_seq_len = self.args_dict["q_seq_len"]
        self.kv_seq_len = self.args_dict["kv_seq_len"]

        self.softmax_scale = self.head_dim ** (-0.5)

        # define max_batch_size and max_seq_len
        self.max_batch_size = self.batch_size
        self.max_seq_len = self.q_seq_len + self.kv_seq_len

        self.total_hidden_dim = (self.q_head_num + 2 * self.kv_head_num) * self.head_dim

        # reference 
        # qkv:          [batch_size, q_seq_len, (q_head_num + 2 * kv_head_num) * head_dim]
        # casual_mask:  [max_seq_len, max_seq_len]
        # key_cache:    [max_batch_size, kv_head_num, max_seq_len, head_dim]
        # value_cache:  [max_batch_size, kv_head_num, max_seq_len, head_dim]
        # kv_len:       [batch_size]
        # kv_idx:       [batch_size]



        



