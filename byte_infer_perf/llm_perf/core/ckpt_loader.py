import os
import sys
import time
import pathlib

import torch
import torch.nn as nn
import torch.distributed as dist

from llm_perf.utils.logger import logger

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from typing import Union, List

class CoreCkptLoader(ABC):
    def __init__(
        self, 
        prefix, model, 
        mp_size=1, mp_rank=0, 
        ckpt_path: str = ""
    ):
        self.prefix = prefix
        self.model = model

        self.mp_size = mp_size
        self.mp_rank = mp_rank

        self.ckpt_path = ckpt_path
        
        self.state_dict = None


    def to_parameter(
        self, 
        data : torch.Tensor, 
        dtype : torch.dtype =None
    ):
        if dtype is not None:
            data = data.to(dtype)
        return nn.Parameter(data, requires_grad=False)


    def to_contiguous(self, num_layers, param_suffixes, prefix, state_dict): 
        result = {}

        with ThreadPoolExecutor() as executor:
            for i in range(num_layers):
                for suffix in param_suffixes:
                    # for example: 
                    # "transformer.encoder.layers.0.mlp.dense_4h_to_h.weight"
                    name = f"{prefix}.{i}.{suffix}"
                    if name in state_dict:
                        result[name] = executor.submit(lambda t : t.contiguous(), state_dict[name])
        
        for i in range(num_layers):
            for suffix in param_suffixes:
                name = f"{prefix}.{i}.{suffix}"
                if name in state_dict:
                    state_dict[name] = result[name].result


    def gqa_split(self, src, dim):
        qkv_head_num = src.shape[dim] // self.head_dim
        src_split = src.chunk(qkv_head_num, dim=dim)
        qkv_cat = []
        for i in range(self.mp_size):
            qkv_cat.append(
                torch.cat(
                    [src_split[i * self.mp_size + self.mp_rank] for i in range(qkv_head_num // self.mp_size)],
                    axis=dim,
                )
            )

        return qkv_cat


    def qkv_split(self, src, dim):
        src_split = torch.split(src.data, src.shape[dim] // 3, dim=dim)
        qkv_split = [torch.split(src_s, src_s.shape[dim] // self.mp_size, dim=dim) for src_s in src_split]
        qkv_cat = [torch.cat([qkv_s[i] for qkv_s in qkv_split], axis=dim) for i in range(len(qkv_split[0]))]
        return qkv_cat
        

    def with_outter_split(
        self, 
        src : torch.Tensor, 
        dim : int, 
        outter : int
    ):
        src_split = torch.split(src.data, src.shape[dim] // outter, dim=dim)
        output_split = [torch.split(src_s, src_s.shape[dim] // self.mp_size, dim=dim) for src_s in src_split]
        output_tensors = [
            torch.cat(
                [output_s[i] for output_s in output_split], 
                axis=dim
            ) for i in range(len(output_split[0]))
        ]
        return output_tensors
        

    def split(
        self, 
        src : torch.Tensor, 
        dim : int, 
        chunks : List [int]=[]
    ):
        if len(chunks) == 0:
            split_arg = src.shape[dim] // self.mp_size
            output_tensors = torch.split(src, split_arg, dim=dim)
        else:
            # for example
            #   chunks = [32, 2, 2], sum_chunks = 36, src.shape[dim] = (32 + 2 + 2) * 128, other_dim = 128
            #   mp_size = 8
            #   new_chunks = [4, 1, 1]
            sum_chunks = sum(chunks)
            other_dim_size = src.shape[dim] // sum_chunks

            split_arg = [i * other_dim_size for i in chunks]            
            split_tensors = torch.split(src, split_arg, dim=dim)

            output_split = []
            for i, tensor in enumerate(split_tensors):
                if self.mp_size > chunks[i]:
                    tensor_shape = tensor.size()[:dim] + (chunks[i], 1, other_dim_size) + tensor.size()[dim+1:]
                    new_tensor_shape = tensor.size()[:dim] + (chunks[i], self.mp_size // chunks[i], other_dim_size) + tensor.size()[dim+1:]
                    output_tensor_shape = tensor.size()[:dim] + (self.mp_size * other_dim_size,) + tensor.size()[dim+1:]

                    tensor = tensor.view(tensor_shape)
                    tensor = tensor.expand(*new_tensor_shape)
                    tensor = tensor.contiguous()
                    tensor = tensor.view(output_tensor_shape)

                cur_split = torch.split(tensor, tensor.shape[dim] // self.mp_size, dim=dim)
                output_split.append(cur_split)

            output_tensors = []
            for i in range(self.mp_size):
                temp_tensors = [output_split[j][i] for j in range(len(chunks))]    
                tp_tensors = torch.concat(temp_tensors, dim=dim)
                output_tensors.append(tp_tensors)

        return output_tensors



    def broadcast_meta(self):
        meta = [
            {k: [v.shape, v.dtype] for k, v in self.state_dict.items()}
        ] if self.mp_rank == 0 else [None]
        dist.broadcast_object_list(meta, src=0)
        dist.barrier()
        if self.mp_rank != 0:
            self.state_dict = {
                k: torch.empty(v[0], dtype=v[1], device='meta') for k, v in meta[0].items()
            }


    @abstractmethod
    def broadcast_weight(self, key, device='cpu', non_blocking=False):
        raise NotImplementedError


    # split_mode
    #   default
    #   with_outter
    #   split_outter
    @abstractmethod
    def scatter_weight(self, key, dim, split_mode='default', outter=1, non_blocking=False):
        raise NotImplementedError


    @abstractmethod
    def parallel_loader(self):
        raise NotImplementedError


    @abstractmethod
    def infusion_to_model(self):
        raise NotImplementedError





    def load(self):
        return self.parallel_loader()


    def torch_load_wrapper(
        self, 
        ckpt_path: str, 
        map_location: Union[str, torch.device]=torch.device('cpu')
    ):
        st = time.time()
        
        state_dict = {}
        model_path = pathlib.Path(ckpt_path)
        if model_path.is_dir():
            if model_path.joinpath("pytorch_model.bin.index.json").exists():
                file_list = []
                for file in model_path.iterdir():
                    if not (file.stem.startswith('pytorch_model-') and file.suffix.endswith('.bin')):
                        continue
                    file_list.append(file)
                file_list.sort()

                for file in file_list:
                    state_dict.update(torch.load(file, map_location=map_location))

        logger.info(f"RANK{self.mp_rank} load {ckpt_path} cost: {time.time() - st}s")

        # for key in state_dict.keys():
        #     print(f"{key}, {state_dict[key].shape}")

        return state_dict
            
                    
        
        
        
        




    