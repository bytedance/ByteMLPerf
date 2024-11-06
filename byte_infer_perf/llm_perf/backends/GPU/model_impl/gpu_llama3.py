import os
import pathlib

import torch
import torch.nn as nn
import torch.distributed as dist

from typing import Dict, Any
from llm_perf.utils.logger import logger
from llm_perf.utils.ps_utils import check_memory_usage
from llm_perf.utils.dist_utils import check_dist

from accelerate import init_empty_weights

from llm_perf.backends.GPU.gpu_ckpt_loader import GpuCkptLoader
from llm_perf.core.ckpt_loader import Llama_ModelLoader
from transformers import LlamaConfig
from .modeling_llama3 import LlamaForCausalLM


class GPULlamaLoader(GpuCkptLoader):
    def __init__(
        self, 
        model : LlamaForCausalLM, 
        model_config : LlamaConfig, 
        ckpt_path : str = ""
    ):
        mp_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        super().__init__("", model, mp_size, local_rank, ckpt_path)
        self.model_config = model_config

    def parallel_loader(self):
        self.state_dict = {}

        model_dir = pathlib.Path(self.ckpt_path).absolute()
        if not model_dir.exists() or not model_dir.is_dir():
            if self.mp_rank == 0:
                print(f"{model_dir} not exists or is not a directory")
            return
        
        split_model_dir = model_dir.joinpath(f"TP{self.mp_size}")
        if not split_model_dir.exists() or not split_model_dir.is_dir():
            if self.mp_rank == 0:
                print(f"{split_model_dir} not exists or is not a directory, please split model first.")
            return

        model_loader = Llama_ModelLoader(split_model_dir / f"device_{self.mp_rank}")
        self.state_dict = model_loader.load_weight()
        
    def infusion_to_model(self):
        self.model.model.embed_tokens.weight = self.to_parameter(self.state_dict["model.embed_tokens.weight"])
        for i in range(self.model_config.num_hidden_layers):
            self.model.model.layers[i].input_layernorm.weight = self.to_parameter(self.state_dict[f"model.layers.{i}.input_layernorm.weight"])

            self.model.model.layers[i].self_attn.q_proj.weight = self.to_parameter(self.state_dict[f"model.layers.{i}.self_attn.q_proj.weight"])
            self.model.model.layers[i].self_attn.k_proj.weight = self.to_parameter(self.state_dict[f"model.layers.{i}.self_attn.k_proj.weight"])
            self.model.model.layers[i].self_attn.v_proj.weight = self.to_parameter(self.state_dict[f"model.layers.{i}.self_attn.v_proj.weight"])
            self.model.model.layers[i].self_attn.o_proj.weight = self.to_parameter(self.state_dict[f"model.layers.{i}.self_attn.o_proj.weight"])

            self.model.model.layers[i].post_attention_layernorm.weight = self.to_parameter(self.state_dict[f"model.layers.{i}.post_attention_layernorm.weight"])

            self.model.model.layers[i].mlp.gate_proj.weight = self.to_parameter(self.state_dict[f"model.layers.{i}.mlp.gate_proj.weight"])
            self.model.model.layers[i].mlp.up_proj.weight = self.to_parameter(self.state_dict[f"model.layers.{i}.mlp.up_proj.weight"])
            self.model.model.layers[i].mlp.down_proj.weight = self.to_parameter(self.state_dict[f"model.layers.{i}.mlp.down_proj.weight"])

        self.model.model.norm.weight = self.to_parameter(self.state_dict["model.norm.weight"])
        self.model.lm_head.weight = self.to_parameter(self.state_dict["lm_head.weight"])


class GPULlama(nn.Module):
    def __init__(self, xpu_cfg: Dict[str, Any]) -> None:
        super().__init__()

        self.xpu_cfg = xpu_cfg
        self.model_config = xpu_cfg["model_config"]

        self.model_name = self.model_config["model_name"]
        self.model_path = self.model_config["model_path"]
        self.model_network = self.model_config["network"]

        self.llama_config : LlamaConfig = LlamaConfig(**self.model_network)
        # print(self.llama_config)

        # dist config
        self.mp_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        self.transformer_model : LlamaForCausalLM = None

    
    def init_inference(self):
        torch.cuda.set_device(self.local_rank)

        if self.mp_size > 1:
            logger.info(f"RANK: {self.local_rank} {self.mp_size} init_process_group...")
            dist.init_process_group(
                backend="nccl",
                world_size=self.mp_size,
                rank=self.local_rank
            )
            check_dist()

        check_memory_usage("Begin")

        with init_empty_weights():
            self.transformer_model = LlamaForCausalLM(self.llama_config).to(self.llama_config.torch_dtype).eval()

        check_memory_usage("After build model")

        self.load_weight(self.model_path)
        
        check_memory_usage("After load_weight")

        self.transformer_model.cuda()

        check_memory_usage("After model to device")

        self.kv_cache = self.init_kvcache(self.llama_config.torch_dtype)

        if self.mp_size > 1:
            dist.barrier()

    def finalize_inference(self):
        if self.mp_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

    def load_weight(self, ckpt_path):
        p_loader = GPULlamaLoader(self.transformer_model, self.llama_config, ckpt_path)
        p_loader.parallel_loader()
        p_loader.infusion_to_model()

    def init_kvcache(self, dtype):
        max_batch_size = self.xpu_cfg["max_batch_size"]
        num_layers = self.llama_config.num_hidden_layers
        max_seq_len = self.llama_config.max_position_embeddings
        hidden_size = self.llama_config.hidden_size
        q_head_num = self.llama_config.num_attention_heads
        kv_head_num = self.llama_config.num_key_value_heads
        head_dim = hidden_size // q_head_num

        cur_device = self.transformer_model.device

        past_key_values = ()
        for i in range(num_layers):
            kv_shape = (max_batch_size, kv_head_num // self.mp_size, max_seq_len, head_dim)
            key_cache = torch.empty(kv_shape, dtype=dtype, device=cur_device)
            value_cache = torch.empty(kv_shape, dtype=dtype, device=cur_device)
            past_key_values += ((key_cache, value_cache),)
        return past_key_values
    

    def forward(self, inputs : Dict[str, torch.Tensor]):
        model_outputs = self.transformer_model.forward(
            **inputs, 
            past_key_values=self.kv_cache
        )

        # context: [1, seq_len] --> [1, seq_len, vocab_size] or [1, 1, vocab_size]
        # decode: [max_batch_size, 1]
        logits = model_outputs.logits

        output_dict = {
            "logits": logits
        }
        return output_dict