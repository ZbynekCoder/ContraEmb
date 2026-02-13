from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor, temp: float) -> torch.Tensor:
    # (N,H) x (M,H) -> (N,M)
    return F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1) / float(temp)


def cos_infonce_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    hard: Optional[torch.Tensor],
    temp: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    InfoNCE (cos) loss:
      logits = [sim(q_i, pos_j), sim(q_i, hard_k)] then CE against diagonal labels.
    Returns (loss, logits).
    """
    logits = cosine_sim_matrix(query, positive, temp)

    if hard is not None:
        logits_hard = cosine_sim_matrix(query, hard, temp)
        logits = torch.cat([logits, logits_hard], dim=1)

        # Keep baseline behavior: it *had* a "z3_weight" branch but z3_weight=0 -> no-op.
        # So we intentionally do nothing here to remain numerically identical.

    labels = torch.arange(logits.size(0), device=device, dtype=torch.long)
    loss = F.cross_entropy(logits, labels)
    return loss, logits


def decouple_loss(
    zq: torch.Tensor,                 # (bs, H)
    zpos: torch.Tensor,               # (bs, H)
    zhard: Optional[torch.Tensor],    # (bs, max_hard, H) or None
    hard_mask: Optional[torch.Tensor],# (bs, max_hard) bool or None
    *,
    temp: float,
    tau_E: Optional[float],
    stance_margin: float,
    stance_beta: float,
    stance_alpha: float,
    stance_query_transform: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Decoupled loss = loss_E + alpha * loss_T

    loss_E: semantic contrast with logsumexp (pos + hard in denom, masked)
    loss_T: stance margin loss using transformed query, stop-grad into encoder
    Returns (loss, logits, aux_dict).
    """
    bs = zq.size(0)
    tau_E_val = float(tau_E) if tau_E is not None else float(temp)

    if zhard is not None and zhard.numel() > 0:
        max_hard = zhard.size(1)
        if hard_mask is None:
            hard_mask = torch.ones((bs, max_hard), dtype=torch.bool, device=zq.device)
        else:
            hard_mask = hard_mask.to(zq.device)

        docs_hard_flat = zhard.reshape(bs * max_hard, -1)
        hard_valid_flat = hard_mask.reshape(bs * max_hard)
    else:
        max_hard = 0
        hard_mask = torch.zeros((bs, 0), dtype=torch.bool, device=zq.device)
        docs_hard_flat = zq.new_zeros((0, zq.size(-1)))
        hard_valid_flat = torch.zeros((0,), dtype=torch.bool, device=zq.device)

    docs_all = torch.cat([zpos, docs_hard_flat], dim=0)  # (bs + bs*max_hard, H)
    doc_valid = torch.cat(
        [torch.ones((bs,), dtype=torch.bool, device=zq.device), hard_valid_flat],
        dim=0
    )  # (bs + bs*max_hard,)

    # logits: (bs, bs + bs*max_hard)
    logits = F.cosine_similarity(zq.unsqueeze(1), docs_all.unsqueeze(0), dim=-1) / tau_E_val

    # positives:
    pos_mask = torch.zeros_like(logits, dtype=torch.bool)
    ar = torch.arange(bs, device=zq.device)
    pos_mask[ar, ar] = True

    if max_hard > 0:
        base = bs + ar.unsqueeze(1) * max_hard
        j = torch.arange(max_hard, device=zq.device).unsqueeze(0)
        hard_indices = base + j
        pos_mask.scatter_(1, hard_indices, hard_mask)

    denom_mask = doc_valid.unsqueeze(0).expand_as(logits)

    neg_inf = torch.finfo(logits.dtype).min
    logits_pos = logits.masked_fill(~pos_mask, neg_inf)
    logits_den = logits.masked_fill(~denom_mask, neg_inf)
    loss_E = -(torch.logsumexp(logits_pos, dim=1) - torch.logsumexp(logits_den, dim=1)).mean()

    # stance loss (only updates stance transform; stopgrad into encoder)
    if max_hard == 0:
        loss_T = zq.new_zeros(())
    else:
        tq = stance_query_transform(zq.detach())  # already normalized inside transform
        sim_pos = F.cosine_similarity(tq, zpos, dim=-1)                   # (bs,)
        sim_hard = F.cosine_similarity(tq.unsqueeze(1), zhard, dim=-1)    # (bs, max_hard)

        delta = (sim_hard - sim_pos.unsqueeze(1)).masked_fill(~hard_mask, neg_inf)
        lse = torch.logsumexp(float(stance_beta) * (delta + float(stance_margin)), dim=1)
        loss_T = (1.0 / float(stance_beta)) * torch.log1p(torch.exp(lse)).mean()

    loss = loss_E + float(stance_alpha) * loss_T
    aux = {"loss_E": loss_E, "loss_T": loss_T}
    return loss, logits, aux


def mlm_aux_loss(
    prediction_scores: torch.Tensor,  # (N, L, vocab)
    mlm_labels: torch.Tensor,         # (N, L)
    vocab_size: int,
) -> torch.Tensor:
    return F.cross_entropy(
        prediction_scores.view(-1, vocab_size),
        mlm_labels.view(-1),
        ignore_index=-100,
    )
