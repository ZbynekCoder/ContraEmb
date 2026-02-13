import json
import logging
import math
import os
from typing import Optional, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    BertLMPredictionHead,
)

from .batching import split_retrieval_batch
from .dist_utils import gather_concat
from .losses import cos_infonce_loss, decouple_loss, mlm_aux_loss

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

        # optional MLM (reuse flat_attn/flat_tti)
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

        # ---- batch organization (extracted) ----
        batch = split_retrieval_batch(pooled, hard_mask)
        q_raw = batch["q_raw"]
        pos = batch["pos"]
        hard_for_cos = batch["hard_for_cos"]      # (bs,H) or None
        hard_all = batch["hard_all"]              # (bs,hn,H) or None
        hard_mask = batch["hard_mask"]            # (bs,hn) or None

        # query transform ONLY affects the cos branch's query, keep decouple semantic on raw q
        q_cos = self._apply_query_transform(q_raw)

        loss_type = str(getattr(self.model_args, "loss_type", "cos")).lower()

        # -------------------------
        # cos branch: DDP gather (single impl; no duplicate all_gather)
        # -------------------------
        if loss_type == "cos":
            # gather across ranks only once (future-proof)
            if dist.is_available() and dist.is_initialized() and self.training:
                if hard_for_cos is not None:
                    hard_for_cos = gather_concat(hard_for_cos)
                q_cos = gather_concat(q_cos)
                pos = gather_concat(pos)

            loss, cos_sim = cos_infonce_loss(
                query=q_cos,
                positive=pos,
                hard=hard_for_cos,
                temp=float(getattr(self.model_args, "temp", 0.05)),
                device=self.device,
            )

            self.custom_epoch_info.setdefault("cl_loss", [])
            self.custom_epoch_info["cl_loss"].append(loss.detach())

        # -------------------------
        # decouple branch: NO DDP gather (same semantics as your current code)
        # -------------------------
        elif loss_type == "decouple":
            tau_E = getattr(self.model_args, "tau_E", None)
            margin = float(getattr(self.model_args, "stance_margin", 0.1))
            beta = float(getattr(self.model_args, "stance_beta", 10.0))
            alpha = float(getattr(self.model_args, "stance_alpha", 1.0))

            def stance_transform(zq_det: torch.Tensor) -> torch.Tensor:
                # same behavior: apply query_transform on detached query then normalize
                if self.use_query_transform and self.query_transform is not None:
                    tq = zq_det + self.query_transform_scale * self.query_transform_dropout(
                        self.query_transform(zq_det)
                    )
                    return F.normalize(tq, p=2, dim=-1)
                return F.normalize(zq_det, p=2, dim=-1)

            loss, cos_sim, aux = decouple_loss(
                zq=q_raw,
                zpos=pos,
                zhard=hard_all,
                hard_mask=hard_mask,
                temp=float(getattr(self.model_args, "temp", 0.05)),
                tau_E=tau_E,
                stance_margin=margin,
                stance_beta=beta,
                stance_alpha=alpha,
                stance_query_transform=stance_transform,
            )

            self.custom_epoch_info.setdefault("cl_loss", [])
            self.custom_epoch_info.setdefault("loss_E", [])
            self.custom_epoch_info.setdefault("loss_T", [])
            self.custom_epoch_info["cl_loss"].append(loss.detach())
            self.custom_epoch_info["loss_E"].append(aux["loss_E"].detach())
            self.custom_epoch_info["loss_T"].append(aux["loss_T"].detach())

        else:
            raise NotImplementedError(f"Undefined loss type: {loss_type}.")

        # MLM loss
        if mlm_outputs is not None and mlm_labels is not None:
            flat_mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
            prediction_scores = self.lm_head(mlm_outputs.last_hidden_state)
            masked_lm_loss = mlm_aux_loss(
                prediction_scores=prediction_scores,
                mlm_labels=flat_mlm_labels,
                vocab_size=self.config.vocab_size,
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
