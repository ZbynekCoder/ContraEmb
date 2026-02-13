import os
import json
import math
import logging
from typing import Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    BertLMPredictionHead,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)

logger = logging.getLogger(__name__)

# =========================
# NaN/Inf debug helper
# =========================
_DEBUG_NAN = os.environ.get("DEBUG_NAN", "0") == "1"
_DEBUG_NAN_MAX_DUMPS = int(os.environ.get("DEBUG_NAN_MAX_DUMPS", "3"))
_debug_nan_dumped = 0


def _finite_check(name: str, t: torch.Tensor, extra: Optional[dict] = None):
    """Raise on NaN/Inf. Optionally dumps a small JSON for repro."""
    global _debug_nan_dumped
    if t is None or (not torch.is_tensor(t)):
        return
    if torch.isfinite(t).all():
        return

    msg = {
        "name": name,
        "dtype": str(t.dtype),
        "shape": list(t.shape),
        "nan_cnt": int(torch.isnan(t).sum().item()),
        "inf_cnt": int(torch.isinf(t).sum().item()),
    }
    if t.numel():
        safe = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        msg["min"] = float(safe.min().item())
        msg["max"] = float(safe.max().item())
    else:
        msg["min"] = None
        msg["max"] = None
    if extra:
        msg.update(extra)

    if _DEBUG_NAN and _debug_nan_dumped < _DEBUG_NAN_MAX_DUMPS:
        out = os.environ.get("DEBUG_NAN_OUT", "./nan_dumps")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"nan_{_debug_nan_dumped:02d}_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(msg, f, ensure_ascii=False, indent=2)
        _debug_nan_dumped += 1

    raise FloatingPointError(f"[NaN/Inf detected] {msg}")


# =========================
# Pooler / Similarity
# =========================
class Pooler(nn.Module):
    """
    'cls': use [CLS] token.
    'cls_before_pooler': same as cls (no extra MLP here).
    'avg': mean pooling with attention_mask.
    'avg_top2': avg of last two layers (requires hidden_states).
    'avg_first_last': avg of first and last layers (requires hidden_states).
    """
    def __init__(self, pooler_type: str):
        super().__init__()
        self.pooler_type = pooler_type
        assert pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"]

    def forward(self, attention_mask: torch.Tensor, outputs) -> torch.Tensor:
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        if self.pooler_type in ("cls", "cls_before_pooler"):
            return last_hidden[:, 0]

        if self.pooler_type == "avg":
            attn = attention_mask.unsqueeze(-1)  # (bs, L, 1)
            return (last_hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp_min(1)

        if self.pooler_type == "avg_first_last":
            first = hidden_states[1]
            last = hidden_states[-1]
            attn = attention_mask.unsqueeze(-1)
            return (((first + last) / 2.0) * attn).sum(dim=1) / attn.sum(dim=1).clamp_min(1)

        if self.pooler_type == "avg_top2":
            second_last = hidden_states[-2]
            last = hidden_states[-1]
            attn = attention_mask.unsqueeze(-1)
            return (((second_last + last) / 2.0) * attn).sum(dim=1) / attn.sum(dim=1).clamp_min(1)

        raise NotImplementedError(self.pooler_type)


class Similarity(nn.Module):
    """Cosine similarity with temperature."""
    def __init__(self, temp: float):
        super().__init__()
        self.temp = float(temp)
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.cos(x, y) / self.temp


# =========================
# Query-side transform (T)
# =========================
class QueryTransform(nn.Module):
    """
    Supports:
      - linear (no bias)
      - mlp
      - gated_mlp (default)
    Initialization is stable: final projection starts at 0 -> near-identity residual update.
    """
    def __init__(self, hidden: int, kind: str = "gated_mlp", mlp_ratio: float = 0.25, init_std: float = 0.02):
        super().__init__()
        kind = str(kind).lower()
        mid = max(1, int(hidden * float(mlp_ratio)))

        self.kind = kind
        self.init_std = float(init_std)

        if kind in ("linear", "lin"):
            self.fc = nn.Linear(hidden, hidden, bias=False)
            nn.init.normal_(self.fc.weight, mean=0.0, std=self.init_std)
        else:
            self.fc1 = nn.Linear(hidden, mid, bias=True)
            self.fc2 = nn.Linear(mid, hidden, bias=False)
            self.act = nn.GELU()
            self.gated = kind not in ("mlp", "fcn")

            if self.gated:
                self.gate = nn.Linear(hidden, mid, bias=True)

            nn.init.normal_(self.fc1.weight, mean=0.0, std=self.init_std)
            nn.init.zeros_(self.fc1.bias)
            if self.gated:
                nn.init.normal_(self.gate.weight, mean=0.0, std=self.init_std)
                nn.init.zeros_(self.gate.bias)

            # key trick: start near-identity (delta≈0 at start)
            nn.init.zeros_(self.fc2.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.kind in ("linear", "lin"):
            return self.fc(z)

        h = self.act(self.fc1(z))
        if self.gated:
            g = torch.sigmoid(self.gate(z))
            h = h * g
        return self.fc2(h)


# =========================
# our_BertForCL (single tower only)
# =========================
class our_BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)

        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        # pooler & sim
        self.pooler_type = self.model_args.pooler_type
        self.pooler = Pooler(self.pooler_type)
        self.sim = Similarity(temp=self.model_args.temp)

        # optional MLM head
        if getattr(self.model_args, "do_mlm", False):
            self.lm_head = BertLMPredictionHead(config)

        # query-side transform (optional)
        self.use_query_transform = bool(getattr(self.model_args, "use_query_transform", False))
        if self.use_query_transform:
            self.query_transform_dropout = nn.Dropout(float(getattr(self.model_args, "query_transform_dropout", 0.1)))
            self.query_transform_scale = float(getattr(self.model_args, "query_transform_scale", 1.0))
            self.query_transform = QueryTransform(
                hidden=config.hidden_size,
                kind=str(getattr(self.model_args, "query_transform_type", "gated_mlp")),
                mlp_ratio=float(getattr(self.model_args, "query_transform_mlp_ratio", 0.25)),
                init_std=float(getattr(self.model_args, "query_transform_init_std", 0.02)),
            )
        else:
            self.query_transform_dropout = None
            self.query_transform_scale = 0.0
            self.query_transform = None

        # training-time tracking for your logger/callback
        self.custom_epoch_info: Dict[str, list] = {"loss": [], "cl_loss": []}

        # misc
        self.hidden_size = config.hidden_size
        self.sqrt_hidden_size = math.sqrt(self.hidden_size)
        self.current_training_progress = 0

        self.post_init()

    # ---------- helpers ----------
    def _need_hidden_states(self) -> bool:
        return self.pooler_type in ("avg_top2", "avg_first_last")

    def _encode_flat(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
    ):
        return self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=self._need_hidden_states(),
            return_dict=True,
        )

    def _apply_query_transform(self, z1: torch.Tensor) -> torch.Tensor:
        if not self.use_query_transform or self.query_transform is None:
            return z1
        z1 = z1 + self.query_transform_scale * self.query_transform_dropout(self.query_transform(z1))
        z1 = F.normalize(z1, p=2, dim=-1)
        return z1

    # ---------- public forwards ----------
    def sentemb_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        return_dict=None,
    ):
        """For inference/encoding: input is (bs, L)."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self._encode_flat(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )
        pooler_output = self.pooler(attention_mask, outputs)

        if not return_dict:
            return (outputs[0], pooler_output) + outputs[2:]
        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )

    def cl_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        return_dict=None,
        mlm_input_ids=None,
        mlm_labels=None,
        hard_mask=None,
    ):
        """
        Training forward: input is (bs, num_sent, L).
          sentence0=query/anchor, sentence1=pos, sentence2.. hard negatives (padded)
        hard_mask: (bs, max_hard) bool mask for valid hard negatives (positions 2..)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bs = input_ids.size(0)
        num_sent = input_ids.size(1)
        seq_len = input_ids.size(-1)

        # flatten
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attn = attention_mask.view(-1, attention_mask.size(-1))
        flat_tti = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        outputs = self._encode_flat(
            input_ids=flat_input_ids,
            attention_mask=flat_attn,
            token_type_ids=flat_tti,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )

        # optional MLM (keep same behavior as original: reuse flat_attn/flat_tti)
        mlm_outputs = None
        if mlm_input_ids is not None:
            flat_mlm_ids = mlm_input_ids.view(-1, mlm_input_ids.size(-1))
            mlm_outputs = self._encode_flat(
                input_ids=flat_mlm_ids,
                attention_mask=flat_attn,
                token_type_ids=flat_tti,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
            )

        # pool then reshape
        pooled = self.pooler(flat_attn, outputs)  # (bs*num_sent, H)
        _finite_check("pooler_output_flat", pooled)
        pooled = pooled.view(bs, num_sent, -1)    # (bs, num_sent, H)

        # split query/positive
        query = pooled[:, 0]
        positive = pooled[:, 1]

        # apply query transform ONLY on query for training
        query = self._apply_query_transform(query)

        # hard negatives (only for cos path's hard_negative gather; decouple uses pooled directly)
        hard_negative = pooled[:, 2]

        # distributed gather for cos loss branch (preserve original behavior)
        if dist.is_initialized() and self.training:
            if num_sent >= 3:
                hard_negative_list = [torch.zeros_like(hard_negative) for _ in range(dist.get_world_size())]
                dist.all_gather(hard_negative_list, hard_negative.contiguous())
                hard_negative_list[dist.get_rank()] = hard_negative
                hard_negative = torch.cat(hard_negative_list, dim=0)

            query_list = [torch.zeros_like(query) for _ in range(dist.get_world_size())]
            positive_list = [torch.zeros_like(positive) for _ in range(dist.get_world_size())]
            dist.all_gather(query_list, query.contiguous())
            dist.all_gather(positive_list, positive.contiguous())
            query_list[dist.get_rank()] = query
            positive_list[dist.get_rank()] = positive
            query = torch.cat(query_list, dim=0)
            positive = torch.cat(positive_list, dim=0)

        loss_type = str(getattr(self.model_args, "loss_type", "cos")).lower()

        # =========================
        # loss: decouple
        # =========================
        if loss_type == "decouple":
            # NOTE: keep semantics close to original: use pooled (local) not gathered query/positive
            zq = pooled[:, 0]       # (bs, H)
            zpos = pooled[:, 1]     # (bs, H)
            zhard = pooled[:, 2:] if num_sent > 2 else None

            tau_E = float(getattr(self.model_args, "tau_E", None) or self.model_args.temp)
            margin = float(getattr(self.model_args, "stance_margin", 0.1))
            beta = float(getattr(self.model_args, "stance_beta", 10.0))
            alpha = float(getattr(self.model_args, "stance_alpha", 1.0))

            if zhard is not None:
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

            docs_all = torch.cat([zpos, docs_hard_flat], dim=0)
            doc_valid = torch.cat(
                [torch.ones((bs,), dtype=torch.bool, device=zq.device), hard_valid_flat],
                dim=0
            )

            logits = nn.CosineSimilarity(dim=-1)(zq.unsqueeze(1), docs_all.unsqueeze(0)) / tau_E

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

            # stance loss (updates T only; stopgrad into encoder)
            if max_hard == 0:
                loss_T = zq.new_zeros(())
            else:
                if self.use_query_transform and self.query_transform is not None:
                    zq_det = zq.detach()
                    tq = zq_det + self.query_transform_scale * self.query_transform_dropout(self.query_transform(zq_det))
                    tq = F.normalize(tq, p=2, dim=-1)
                else:
                    tq = F.normalize(zq.detach(), p=2, dim=-1)

                sim_pos = nn.CosineSimilarity(dim=-1)(tq, zpos)
                sim_hard = nn.CosineSimilarity(dim=-1)(tq.unsqueeze(1), zhard)

                delta = (sim_hard - sim_pos.unsqueeze(1)).masked_fill(~hard_mask, neg_inf)
                lse = torch.logsumexp(beta * (delta + margin), dim=1)
                loss_T = (1.0 / beta) * torch.log1p(torch.exp(lse)).mean()

            loss = loss_E + alpha * loss_T

            if self.custom_epoch_info is None:
                self.custom_epoch_info = {}
            self.custom_epoch_info.setdefault("cl_loss", [])
            self.custom_epoch_info.setdefault("loss_E", [])
            self.custom_epoch_info.setdefault("loss_T", [])
            self.custom_epoch_info["cl_loss"].append(loss.detach())
            self.custom_epoch_info["loss_E"].append(loss_E.detach())
            self.custom_epoch_info["loss_T"].append(loss_T.detach())

            cos_sim = logits  # keep for compatibility

        # =========================
        # loss: cos (InfoNCE)
        # =========================
        elif loss_type == "cos":
            cos_sim = self.sim(query.unsqueeze(1), positive.unsqueeze(0))
            if num_sent >= 3:
                z1_z3_cos = self.sim(query.unsqueeze(1), hard_negative.unsqueeze(0))
                cos_sim = torch.cat([cos_sim, z1_z3_cos], dim=1)

            labels = torch.arange(cos_sim.size(0)).long().to(self.device)
            loss_fct = nn.CrossEntropyLoss()

            # keep EXACT baseline behavior: hard_negative_weight hard-coded as 0 in original
            if num_sent == 3:
                hard_negative_weight = 0
                weights = torch.tensor(
                    [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [hard_negative_weight] + [0.0] * (
                        z1_z3_cos.size(-1) - i - 1)
                     for i in range(z1_z3_cos.size(-1))]
                ).to(self.device)
                cos_sim = cos_sim + weights

            loss = loss_fct(cos_sim, labels)
            self.custom_epoch_info.setdefault("cl_loss", [])
            self.custom_epoch_info["cl_loss"].append(loss.detach())

        else:
            raise NotImplementedError(f"Undefined loss type: {loss_type}.")

        # MLM loss
        if mlm_outputs is not None and mlm_labels is not None:
            flat_mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
            prediction_scores = self.lm_head(mlm_outputs.last_hidden_state)
            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size),
                flat_mlm_labels.view(-1),
                ignore_index=-100,
            )
            loss = loss + float(getattr(self.model_args, "mlm_weight", 0.1)) * masked_lm_loss

        if not return_dict:
            output = (cos_sim,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=cos_sim,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb: bool = False,
        mlm_input_ids=None,
        mlm_labels=None,
        hard_mask=None,
        # keep signature tolerant: some callers might still pass is_query from old dual code
        is_query: Optional[bool] = None,
        **kwargs,
    ):
        if sent_emb:
            return self.sentemb_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
        return self.cl_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            return_dict=return_dict,
            mlm_input_ids=mlm_input_ids,
            mlm_labels=mlm_labels,
            hard_mask=hard_mask,
        )
