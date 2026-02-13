import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


@torch.no_grad()
def gather_concat(t: torch.Tensor) -> torch.Tensor:
    """
    All-gather a tensor across ranks and concat on dim=0.
    - If not in DDP, return t unchanged.
    - Preserves dtype/device.
    """
    if not is_dist():
        return t

    world = dist.get_world_size()
    out = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(out, t.contiguous())
    out[dist.get_rank()] = t
    return torch.cat(out, dim=0)
