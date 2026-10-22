"""LLM op: qk_rms_norm (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("qk_rms_norm", "ComputeEngine")
class QKRMSNormOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type not in ["llm"]:
            raise ValueError(
                f"QKRMSNormOp only support llm arg_type, but got {self.arg_type}"
            )

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.qk_head_dim = self.args_dict["qk_head_dim"]
        self.v_head_dim = self.args_dict.get("v_head_dim", self.qk_head_dim)
        
        self.eps = 1e-5

        # 以下参数决定当前 qk_rms_norm 的具体数据类型
        self.dtype = self.args_dict["dtype"]

    def vendor_parser(self):
        if self.dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(
                f"QKRMSNormOp only supports float32, float16, bfloat16, got dtype={self.dtype}"
            )

    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)

        norm_dim = (self.q_head_num + self.kv_head_num) * self.qk_head_dim
        not_norm_dim = self.kv_head_num * self.v_head_dim
        total_dim = norm_dim + not_norm_dim

        # in-place
        self.input_tensor_info = {}
        self.output_tensor_info = {}

        self.input_tensor_info["token_data"] = OpTensorInfo(
            shape=[self.num_tokens, total_dim],
            dtype=self.torch_dtype,
            device=self.backend.get_torch_device_name(),
        )
        self.input_tensor_info["q_norm_weight"] = OpTensorInfo(
            shape=[self.qk_head_dim, ],
            dtype=torch.float32,
            device=self.backend.get_torch_device_name(),
            creator=torch.ones
        )
        self.input_tensor_info["k_norm_weight"] = OpTensorInfo(
            shape=[self.qk_head_dim, ],
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
            calc_tensor_size(self.input_tensor_info["token_data"]) / total_dim * norm_dim
        self.write_bytes = self.read_bytes
        self.read_bytes += calc_tensor_size(self.input_tensor_info["q_norm_weight"])
        self.read_bytes += calc_tensor_size(self.input_tensor_info["k_norm_weight"])
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
        q_norm_weight = tensor_mapping["q_norm_weight"]
        k_norm_weight = tensor_mapping["k_norm_weight"]

        # in-place norm on specified heads
        q_start = 0
        q_end = self.q_head_num * self.qk_head_dim
        q_head_data = token_data[:, q_start:q_end]
        q_head_data = q_head_data.view(-1, self.q_head_num, self.qk_head_dim)

        k_start = self.q_head_num * self.qk_head_dim
        k_end = k_start + self.kv_head_num * self.qk_head_dim
        k_head_data = token_data[:, k_start:k_end]
        k_head_data = k_head_data.view(-1, self.kv_head_num, self.qk_head_dim)

        q_head_data.copy_(
            torch.nn.functional.rms_norm(
                q_head_data,
                normalized_shape=q_head_data.shape[-1:],
                weight=q_norm_weight,
                eps=self.eps,
            )
        )
        k_head_data.copy_(
            torch.nn.functional.rms_norm(
                k_head_data,
                normalized_shape=k_head_data.shape[-1:],
                weight=k_norm_weight,
                eps=self.eps,
            )
        )

        return token_data


