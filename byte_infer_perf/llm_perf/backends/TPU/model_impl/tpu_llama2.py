import os
import pathlib

import torch
import torch.nn as nn
import torch.distributed as dist
import torch_tpu

from typing import Dict, Any
from llm_perf.utils.logger import logger
from llm_perf.utils.ps_utils import check_memory_usage
from llm_perf.utils.dist_utils import check_dist

from accelerate import init_empty_weights

from llm_perf.core.ckpt_loader import Llama_ModelLoader
from transformers import LlamaConfig
# from .tpu_llama2 import 

