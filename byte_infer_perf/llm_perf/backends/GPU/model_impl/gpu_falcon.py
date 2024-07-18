import os
import json
import pathlib

import torch
import torch.distributed as dist
import torch.nn as nn

from typing import Dict, Any
from llm_perf.utils.logger import logger
from llm_perf.utils.ps_utils import check_memory_usage
from llm_perf.utils.dist_utils import check_dist

from accelerate import init_empty_weights

from llm_perf.core.ckpt_loader import CoreCkptLoader, Falcon_ModelLoader
from llm_perf.backends.GPU.gpu_ckpt_loader import GpuCkptLoader
from transformers import FalconConfig
from .falcon import FalconForCausalLM


class GPUFalconLoader(GpuCkptLoader):
    def __init__(
        self, 
        prefix, 
        model, model_config, 
        mp_size=1, mp_rank=0, 
        ckpt_path=None,
    ):
        super().__init__(prefix, model, mp_size, mp_rank, ckpt_path)

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

        model_loader = Falcon_ModelLoader(split_model_dir / f"device_{self.mp_rank}")
        self.state_dict = model_loader.load_weight()

    def infusion_to_model(self):
        pass



class GPUFalcon(nn.Module):
    def __init__(self, xpu_cfg: Dict[str, Any]) -> None:
        super().__init__()

        self.xpu_cfg = xpu_cfg
        self.model_config = xpu_cfg["model_config"]

        self.model_name = self.model_config["model_name"]
        self.model_path = self.model_config["model_path"]
        self.model_network = self.model_config["network"]

        self.falcon_config = FalconConfig(**self.model_network)
        # print(self.falcon_config)

        # dist config
        self.mp_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        self.prefix = ""
        self.transformer_model : FalconForCausalLM = None


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
            self.transformer_model = FalconForCausalLM(
                self.falcon_config
            )
            self.transformer_model.eval()

        self.load_weight(self.model_path)

    def load_weight(self, ckpt_path):
        p_loader = GPUFalconLoader(
            self.prefix, self.transformer_model, self.falcon_config, 
            self.mp_size, self.local_rank, 
            ckpt_path
        )
        p_loader.parallel_loader()
        p_loader.infusion_to_model()


    def forward(self, inputs : Dict[str, torch.Tensor]):
        pass