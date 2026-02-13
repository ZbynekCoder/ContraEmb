from __future__ import annotations

from typing import Dict, Optional

import torch


def split_retrieval_batch(
    pooled: torch.Tensor,
    hard_mask: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Standardize batch layout for retrieval CL training.

    Inputs:
      pooled: (bs, num_sent, H)
        - pooled[:, 0] is query/anchor
        - pooled[:, 1] is positive
        - pooled[:, 2:] are hard negatives (padded)
      hard_mask: (bs, max_hard) bool or None
        - valid hard negatives for pooled[:, 2:]

    Outputs dict keys:
      - bs, num_sent: int tensors (scalar on device)
      - q_raw: (bs, H)
      - pos: (bs, H)
      - hard_for_cos: (bs, H) or None  (keeps baseline behavior: only uses the FIRST hard)
      - hard_all: (bs, hn, H) or None  (all hards for decouple)
      - hard_mask: (bs, hn) bool or None
    """
    if pooled.dim() != 3:
        raise ValueError(f"pooled must be 3D (bs,num_sent,H), got shape={tuple(pooled.shape)}")

    bs, num_sent, h = pooled.shape
    if num_sent < 2:
        raise ValueError(f"Need at least 2 sentences [query, positive], got num_sent={num_sent}")

    q_raw = pooled[:, 0]
    pos = pooled[:, 1]

    if num_sent >= 3:
        hard_for_cos = pooled[:, 2]          # baseline: only the first hard is used in cos loss
        hard_all = pooled[:, 2:]             # (bs, hn, H)
        hn = hard_all.size(1)

        if hard_mask is not None:
            if hard_mask.dim() != 2:
                raise ValueError(f"hard_mask must be 2D (bs,hn), got shape={tuple(hard_mask.shape)}")
            if hard_mask.size(0) != bs:
                raise ValueError(f"hard_mask bs mismatch: {hard_mask.size(0)} vs {bs}")
            if hard_mask.size(1) != hn:
                raise ValueError(f"hard_mask hn mismatch: {hard_mask.size(1)} vs {hn}")
            hard_mask = hard_mask.to(device=pooled.device, dtype=torch.bool)
        # if hard_mask is None: we keep None (and let decouple_loss decide how to treat it)
    else:
        hard_for_cos = None
        hard_all = None
        hard_mask = None

    return {
        "bs": torch.tensor(bs, device=pooled.device),
        "num_sent": torch.tensor(num_sent, device=pooled.device),
        "q_raw": q_raw,
        "pos": pos,
        "hard_for_cos": hard_for_cos,
        "hard_all": hard_all,
        "hard_mask": hard_mask,
    }
