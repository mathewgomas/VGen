# utils/distributed.py
import torch
import torch.distributed as dist
import os

def init_distributed(args):
    """
    Initialize the distributed environment.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"Initializing distributed training on rank {rank} with world size {world_size}")
        torch.distributed.init_process_group(
            backend="nccl",  # Or "gloo" if you're using Gloo
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
    else:
        print("Not initializing distributed training")

def get_rank():
    """
    Returns the rank of the current process.
    """
    if is_dist_initialized():
        return dist.get_rank()
    else:
        return 0

def is_dist_initialized():
    """
    Returns True if the distributed environment is initialized.
    """
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    """
    Returns True if this process is the main process.
    """
    return get_rank() == 0

def synchronize():
    """
    Synchronizes all processes.
    """
    if not is_dist_initialized():
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    dist.barrier()
import torch

def generalized_all_gather(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return tensors
    
    tensor_list = [[] for _ in range(world_size)]
    for tensor in tensors:
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, tensor)
        tensor_list = [t_list + [t] for t_list, t in zip(tensor_list, gathered)]
    
    return [torch.cat(t_list, dim=0) for t_list in tensor_list]
import warnings
def all_reduce(tensor):
    """
    Performs an all-reduce operation on the provided tensor.
    """
    if not is_dist_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

warnings.filterwarnings("ignore", module="torch.cuda")
warnings.filterwarnings("ignore", module="xformers.ops.fmha.flash")
warnings.filterwarnings("ignore", module="rotary_embedding_torch.rotary_embedding_torch")
