"""LLM op: head_rms_norm (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("head_rms_norm", "ComputeEngine")
class HeadRMSNormOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type not in ["llm"]:
            raise ValueError(
                f"HeadRMSNormOp only support llm arg_type, but got {self.arg_type}"
            )

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.total_head_num = self.args_dict["total_head_num"]
        self.head_dim = self.args_dict["head_dim"]

        self.norm_head_start = self.args_dict["norm_head_start"]
        self.norm_head_num = self.args_dict["norm_head_num"]
        self.norm_head_end = self.norm_head_start + self.norm_head_num

        self.eps = 1e-5

        # 以下参数决定当前 head_rms_norm 的具体数据类型
        self.dtype = self.args_dict["dtype"]


    def vendor_parser(self):
        if self.dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(
                f"HeadRMSNormOp only supports float32, float16, bfloat16, got dtype={self.dtype}"
            )

    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)

        # in-place
        self.input_tensor_info = {}
        self.output_tensor_info = {}

        self.input_tensor_info["token_data"] = OpTensorInfo(
            shape=[
                self.num_tokens, 
                self.total_head_num,
                self.head_dim
            ],
            dtype=self.torch_dtype,
            device=self.backend.get_torch_device_name(),
        )
        self.input_tensor_info["norm_weight"] = OpTensorInfo(
            shape=[self.head_dim, ],
            dtype=torch.float32,
            device=self.backend.get_torch_device_name(),
            creator=torch.ones
        )

        # calculator
        self.input_tensor_size = sum(
            [calc_tensor_size(info) for info in self.input_tensor_info.values()]
        )
        self.output_tensor_size = sum(
            [calc_tensor_size(info) for info in self.output_tensor_info.values()]
        )
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = \
            calc_tensor_size(self.input_tensor_info["token_data"]) \
            / self.total_head_num \
            * self.norm_head_num
        self.write_bytes = self.read_bytes
        self.read_bytes += calc_tensor_size(self.input_tensor_info["norm_weight"])
        self.io_bytes = self.read_bytes + self.write_bytes

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=False
        )
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        # get pre-allocated input tensors
        token_data = tensor_mapping["token_data"]
        norm_weight = tensor_mapping["norm_weight"]

        # in-place norm on specified heads
        head_data = token_data[:, self.norm_head_start:self.norm_head_end, :]
        head_data.copy_(
            torch.nn.functional.rms_norm(
                head_data,
                normalized_shape=head_data.shape[-1:],
                weight=norm_weight,
                eps=self.eps,
            )
        )

        return token_data


