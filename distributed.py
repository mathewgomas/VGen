import torch
import torch.distributed as dist

def generalized_all_gather(tensor, world_size, group=None, sync_grads=False):
    """
    Gathers tensors from all processes in a distributed group.

    Args:
        tensor: The tensor to gather.
        world_size: The number of processes in the group.
        group: The process group to gather from.
        sync_grads: Whether to synchronize gradients before gathering.

    Returns:
        A list of tensors gathered from all processes.
    """
    if sync_grads:
        dist.barrier(group=group)
    
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=group)
    return tensor_list

def all_reduce(tensor, group=None, sync_grads=False):
    """
    Reduces a tensor across all processes in a distributed group.

    Args:
        tensor: The tensor to reduce.
        group: The process group to reduce across.
        sync_grads: Whether to synchronize gradients before reducing.

    Returns:
        The reduced tensor.
    """
    if sync_grads:
        dist.barrier(group=group)
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor
