import torch
import torch.distributed as dist

from llm_perf.core.ckpt_loader import CoreCkptLoader

class GpuCkptLoader(CoreCkptLoader):
    def __init__(
        self, 
        prefix, model, 
        mp_size=1, mp_rank=0, 
        ckpt_path: str=""
    ):
        super().__init__(prefix, model, mp_size, mp_rank, ckpt_path)


    def weight_to_device(self, weight : torch.Tensor, non_blocking=False):
        if self.mp_rank == 0:
            weight = weight.cuda(non_blocking=non_blocking)
        else:
            cur_device = torch.cuda.current_device()
            weight = torch.empty_like(weight, device=f"cuda:{cur_device}")
        return weight
    
    def broadcast_weight(self, key, device='cpu', non_blocking=False):
        weight = self.weight_to_device(self.state_dict[key])
        dist.broadcast(weight, src=0)
        dist.barrier()
        self.state_dict[key] = weight.to(device, non_blocking=non_blocking)

    def scatter_weight(self, key, dim, split_mode='default', outter=1, device='cpu', non_blocking=False):
        self.broadcast_weight(key, 'cuda')
        weight = self.state_dict[key]

        if split_mode == 'default':
            weight_split = self.split(weight, dim)
        elif split_mode == 'with_outter':
            weight_split = self.with_outter_split(weight, dim, outter)
        elif split_mode == 'split_outter':
            weight_split = self.split(weight, dim, outter)
        else:
            assert False, f"unknown split mode {split_mode}"


        weight_split = [x.contiguous() for x in weight_split]
        weight = weight_split[self.mp_rank].clone()
        self.state_dict[key] = weight.to(device, non_blocking=non_blocking)