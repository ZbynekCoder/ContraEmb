import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead, BertEncoder, \
    BertLayer, BertEmbeddings, BertPooler
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions, \
    BaseModelOutputWithPastAndCrossAttentions
import math
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class OurBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = config.gradient_checkpointing if hasattr(config,
                                                                               'gradient_checkpointing') else False

    def _gradient_checkpointing_func(self, layer_call, *args, **kwargs):
        """
        A wrapper function for performing gradient checkpointing on a single layer.
        The `layer_call` is a reference to the layer's `__call__` method.
        """
        # `checkpoint` function only accepts tensors as inputs, so filter args to remove None or other non-tensor objects
        tensor_args = tuple(arg for arg in args if isinstance(arg, torch.Tensor))
        # Use the `checkpoint` function from PyTorch, pass the callable (layer_call), followed by the arguments it needs
        return checkpoint(layer_call, *tensor_args, **kwargs)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_args = (
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module, *layer_args
                )
            else:
                layer_outputs = layer_module(*layer_args)

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class OurBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = OurBertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2",
                                    "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)

    cls.init_weights()

    cls.hidden_size = config.hidden_size
    cls.sqrt_hidden_size = math.sqrt(cls.hidden_size)
    cls.current_training_progress = 0
    print(f"cls.hidden_size: {cls.hidden_size}, cls.sqrt_hidden_size: {cls.sqrt_hidden_size}")

    # ====== Query-side learnable transform T (only affects anchor/query z1) ======
    cls.use_query_transform = getattr(cls.model_args, "use_query_transform", False)
    if cls.use_query_transform:
        cls.query_transform_dropout = nn.Dropout(getattr(cls.model_args, "query_transform_dropout", 0.1))
        cls.query_transform_scale = float(getattr(cls.model_args, "query_transform_scale", 1.0))

        # Upgrade: linear / mlp / gated_mlp (default)
        hidden = config.hidden_size
        mlp_ratio = float(getattr(cls.model_args, "query_transform_mlp_ratio", 0.25))  # 0.25/0.5/1.0
        mid = max(1, int(hidden * mlp_ratio))

        qt_type = str(getattr(cls.model_args, "query_transform_type", "gated_mlp")).lower()

        class _QueryMLP(nn.Module):
            def __init__(self, hidden_size: int, mid_size: int, gated: bool):
                super().__init__()
                self.gated = gated
                self.fc1 = nn.Linear(hidden_size, mid_size, bias=True)
                self.fc2 = nn.Linear(mid_size, hidden_size, bias=False)
                if gated:
                    self.gate = nn.Linear(hidden_size, mid_size, bias=True)
                self.act = nn.GELU()

                # init for stability
                init_std = float(getattr(cls.model_args, "query_transform_init_std", 0.02))
                nn.init.normal_(self.fc1.weight, mean=0.0, std=init_std)
                nn.init.zeros_(self.fc1.bias)

                if gated:
                    nn.init.normal_(self.gate.weight, mean=0.0, std=init_std)
                    nn.init.zeros_(self.gate.bias)

                # key trick: start near-identity (delta≈0 at start)
                nn.init.zeros_(self.fc2.weight)

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                h = self.act(self.fc1(z))
                if self.gated:
                    g = torch.sigmoid(self.gate(z))
                    h = h * g
                return self.fc2(h)

        if qt_type in ["linear", "lin"]:
            cls.query_transform = nn.Linear(hidden, hidden, bias=False)
            init_std = float(getattr(cls.model_args, "query_transform_init_std", 0.02))
            nn.init.normal_(cls.query_transform.weight, mean=0.0, std=init_std)

        elif qt_type in ["mlp", "fcn"]:
            cls.query_transform = _QueryMLP(hidden, mid, gated=False)

        else:
            # default: gated_mlp
            cls.query_transform = _QueryMLP(hidden, mid, gated=True)
    # ====== End query transform ======


def our_cl_forward(cls,
                   encoder,
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
                   mlm_input_ids=None,
                   mlm_labels=None,
                   hard_mask=None,
                   ):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    # if cls.pooler_type == "cls":
    # pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

    # ====== Apply learnable transform T on anchor/query only ======
    if getattr(cls.model_args, "use_query_transform", False):
        # Residual transform: T(z) = normalize(z + scale * Dropout(Wz))
        scale = float(getattr(cls.model_args, "query_transform_scale", 1.0))
        if hasattr(cls, "query_transform"):
            z1 = z1 + scale * cls.query_transform_dropout(cls.query_transform(z1))
            z1 = F.normalize(z1, p=2, dim=-1)
    # ====== End transform ======

    # Hard negative
    if num_sent > 2:
        z3 = pooler_output[:, 2]

    if num_sent > 3:
        z4 = pooler_output[:, 3]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    loss_type = cls.model_args.loss_type

    # print(loss_type)

    if loss_type == "decouple":
        # =========================
        # Decoupled training:
        #   - Semantic loss L_E: positives = {d+} ∪ hard negatives; negatives = all other docs in batch
        #   - Stance loss  L_T:   distinguish d+ vs hard negatives using a query-side transform T,
        #                        with gradients blocked from T branch into encoder E (stopgrad).
        # Inputs:
        #   input_ids:   (bs, num_sent, L) where sentence0=q, sentence1=d+, sentence2..=hard negs (padded)
        #   hard_mask:   (bs, max_hard) bool mask for valid hard negatives (positions 2..)
        # =========================
        bs = batch_size
        # split embeddings
        zq = pooler_output[:, 0]                 # (bs, h)
        zpos = pooler_output[:, 1]               # (bs, h)
        zhard = pooler_output[:, 2:] if num_sent > 2 else None   # (bs, max_hard, h)

        # -------- Semantic loss L_E (updates E) --------
        tau_E = float(getattr(cls.model_args, "tau_E", None) or cls.model_args.temp)
        # docs_all = [all positives] + [all hard negatives (padded entries masked out)]
        docs_pos = zpos  # (bs, h)
        if zhard is not None:
            max_hard = zhard.size(1)
            if hard_mask is None:
                # if not provided, assume all are valid
                hard_mask = torch.ones((bs, max_hard), dtype=torch.bool, device=zq.device)
            else:
                hard_mask = hard_mask.to(zq.device)
            docs_hard_flat = zhard.reshape(bs * max_hard, -1)  # (bs*max_hard, h)
            hard_valid_flat = hard_mask.reshape(bs * max_hard)  # (bs*max_hard,)
        else:
            max_hard = 0
            hard_mask = torch.zeros((bs, 0), dtype=torch.bool, device=zq.device)
            docs_hard_flat = zq.new_zeros((0, zq.size(-1)))
            hard_valid_flat = torch.zeros((0,), dtype=torch.bool, device=zq.device)

        docs_all = torch.cat([docs_pos, docs_hard_flat], dim=0)  # (Ndocs, h)
        doc_valid = torch.cat([torch.ones((bs,), dtype=torch.bool, device=zq.device), hard_valid_flat], dim=0)  # (Ndocs,)

        # logits: (bs, Ndocs)
        logits = nn.CosineSimilarity(dim=-1)(zq.unsqueeze(1), docs_all.unsqueeze(0)) / tau_E

        # positive mask: each query i has positives at:
        #   - its own positive doc at index i
        #   - its own valid hard negatives at indices bs + i*max_hard + j
        pos_mask = torch.zeros_like(logits, dtype=torch.bool)  # (bs, Ndocs)
        ar = torch.arange(bs, device=zq.device)
        pos_mask[ar, ar] = True
        if max_hard > 0:
            base = bs + ar.unsqueeze(1) * max_hard  # (bs,1)
            j = torch.arange(max_hard, device=zq.device).unsqueeze(0)  # (1,max_hard)
            hard_indices = base + j  # (bs,max_hard)
            # apply hard_mask
            pos_mask.scatter_(1, hard_indices, hard_mask)

        # denom mask excludes padded hard negatives
        denom_mask = doc_valid.unsqueeze(0).expand_as(logits)

        # L_E = -(logsumexp(pos) - logsumexp(denom))
        neg_inf = torch.finfo(logits.dtype).min
        logits_pos = logits.masked_fill(~pos_mask, neg_inf)
        logits_den = logits.masked_fill(~denom_mask, neg_inf)
        lse_pos = torch.logsumexp(logits_pos, dim=1)
        lse_den = torch.logsumexp(logits_den, dim=1)
        loss_E = -(lse_pos - lse_den).mean()

        # -------- Stance loss L_T (updates T only; stopgrad into E) --------
        margin = float(getattr(cls.model_args, "stance_margin", 0.1))
        beta = float(getattr(cls.model_args, "stance_beta", 10.0))
        alpha = float(getattr(cls.model_args, "stance_alpha", 1.0))

        if max_hard == 0:
            loss_T = zq.new_zeros(())
        else:
            # T(z) = normalize(z + scale*Dropout(Wz)) on detached zq
            if getattr(cls.model_args, "use_query_transform", False) and hasattr(cls, "query_transform"):
                scale = float(getattr(cls.model_args, "query_transform_scale", 1.0))
                zq_det = zq.detach()
                tq = zq_det + scale * cls.query_transform_dropout(cls.query_transform(zq_det))
                tq = F.normalize(tq, p=2, dim=-1)
            else:
                tq = F.normalize(zq.detach(), p=2, dim=-1)

            sim_pos = nn.CosineSimilarity(dim=-1)(tq, zpos)  # (bs,)
            sim_hard = nn.CosineSimilarity(dim=-1)(tq.unsqueeze(1), zhard)  # (bs,max_hard)

            delta = sim_hard - sim_pos.unsqueeze(1)  # (bs,max_hard)
            delta = delta.masked_fill(~hard_mask, neg_inf)

            lse = torch.logsumexp(beta * (delta + margin), dim=1)  # (bs,)
            loss_T = (1.0 / beta) * torch.log1p(torch.exp(lse)).mean()

        loss = loss_E + alpha * loss_T
        if not hasattr(cls, "custom_epoch_info") or cls.custom_epoch_info is None:
            cls.custom_epoch_info = {"cl_loss": [], "loss_E": [], "loss_T": []}
        for k in ["loss_E", "loss_T"]:
            if k not in cls.custom_epoch_info:
                cls.custom_epoch_info[k] = []
        cls.custom_epoch_info["cl_loss"].append(loss.detach())
        cls.custom_epoch_info["loss_E"].append(loss_E.detach())
        cls.custom_epoch_info["loss_T"].append(loss_T.detach())

        cos_sim = logits  # keep a logits tensor for compatibility/logging


    elif loss_type == "cos":
        cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        # Hard negative
        if num_sent >= 3:
            z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
        loss_fct = nn.CrossEntropyLoss()

        # Calculate loss with hard negatives
        if num_sent == 3:
            # Note that weights are actually logits of weights
            z3_weight = 0
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (
                            z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(cls.device)
            cos_sim = cos_sim + weights

        loss = loss_fct(cos_sim, labels)
        cls.custom_epoch_info["cl_loss"].append(loss)

    else:
        raise NotImplementedError(
            f"Undefined loss type: {loss_type}."
        )

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = F.cross_entropy(
            prediction_scores.view(-1, cls.config.vocab_size),
            mlm_labels.view(-1),
            ignore_index=-100,
        )
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
        cls,
        encoder,
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
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    # if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
    # pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class our_BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)
        self.custom_epoch_info = {
            "loss": [],
            "cl_loss": []
        }
        cl_init(self, config)

    def forward(self,
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
                sent_emb=False,
                mlm_input_ids=None,
                mlm_labels=None,
                hard_mask=None,
                ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds,
                                   labels=labels,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict,
                                   )
        else:
            return our_cl_forward(self, self.bert,
                                  input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  head_mask=head_mask,
                                  inputs_embeds=inputs_embeds,
                                  labels=labels,
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states,
                                  return_dict=return_dict,
                                  mlm_input_ids=mlm_input_ids,
                                  mlm_labels=mlm_labels,
                                  hard_mask=hard_mask,
                                  )


# =========================
# Dual Encoder (Route C): doc tower frozen, query tower finetuned
# =========================

class DualBertForCL(BertPreTrainedModel):
    """
    Asymmetric dual-tower bi-encoder for contrastive learning:
      - query_bert: trainable (finetune)
      - doc_bert: frozen (no grad)
    Input format is SAME as current training pipeline:
      input_ids: (bs, num_sent, L) where:
        sentence0 = query/anchor
        sentence1 = positive doc
        sentence2.. = hard negatives (optional)
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]

        # mark in config for auto-detection in eval
        try:
            self.config.dual_encoder = True
        except Exception:
            pass

        # two towers
        self.query_bert = BertModel(config, add_pooling_layer=False)
        self.doc_bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        # tracking (keep same keys as your current model for trainer logging)
        self.custom_epoch_info = {
            "loss": [],
            "cl_loss": []
        }

        # init pooler/sim/query_transform etc (reuse your existing logic)
        cl_init(self, config)

        self.post_init()

    def _encode_with(self, encoder, input_ids, attention_mask, token_type_ids=None,
                     position_ids=None, head_mask=None, inputs_embeds=None,
                     output_attentions=None, output_hidden_states=None, return_dict=True):
        return encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if self.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    def _sentemb_forward_with(self, encoder, input_ids=None, attention_mask=None,
                              token_type_ids=None, position_ids=None, head_mask=None,
                              inputs_embeds=None, output_attentions=None,
                              output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self._encode_with(
            encoder,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        pooler_output = self.pooler(attention_mask, outputs)
        if not return_dict:
            return (outputs[0], pooler_output) + outputs[2:]
        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )

    def _dual_cl_forward(
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
            mlm_input_ids=None,
            mlm_labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)
        assert num_sent >= 2, "Dual encoder needs at least (query, pos_doc)."

        # split query (sent0) vs docs (sent1..)
        q_input_ids = input_ids[:, 0, :]
        q_attn = attention_mask[:, 0, :]
        q_tti = token_type_ids[:, 0, :] if token_type_ids is not None else None

        d_input_ids = input_ids[:, 1:, :].contiguous().view(-1, input_ids.size(-1))
        d_attn = attention_mask[:, 1:, :].contiguous().view(-1, attention_mask.size(-1))
        d_tti = token_type_ids[:, 1:, :].contiguous().view(-1, token_type_ids.size(
            -1)) if token_type_ids is not None else None

        # ---- query tower (trainable) ----
        q_outputs = self._encode_with(
            self.query_bert,
            input_ids=q_input_ids,
            attention_mask=q_attn,
            token_type_ids=q_tti,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        zq = self.pooler(q_attn, q_outputs)  # (bs, H)

        # optional query-side transform (still supported)
        if getattr(self.model_args, "use_query_transform", False):
            scale = float(getattr(self.model_args, "query_transform_scale", 1.0))
            if hasattr(self, "query_transform") and self.query_transform is not None:
                zq = zq + scale * self.query_transform_dropout(self.query_transform(zq))
                zq = F.normalize(zq, p=2, dim=-1)

        # ---- doc tower (frozen) ----
        # If doc tower is frozen, using no_grad saves memory and keeps it strictly anchor space.
        doc_frozen = bool(getattr(self.model_args, "freeze_doc_encoder", True))
        if doc_frozen:
            with torch.no_grad():
                d_outputs = self._encode_with(
                    self.doc_bert,
                    input_ids=d_input_ids,
                    attention_mask=d_attn,
                    token_type_ids=d_tti,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )
        else:
            d_outputs = self._encode_with(
                self.doc_bert,
                input_ids=d_input_ids,
                attention_mask=d_attn,
                token_type_ids=d_tti,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        zd_all = self.pooler(d_attn, d_outputs)  # (bs*(num_sent-1), H)
        zd_all = zd_all.view(batch_size, num_sent - 1, -1)
        zp = zd_all[:, 0, :]  # positive doc
        if num_sent > 2:
            z3 = zd_all[:, 1, :]  # first hard neg (optional)
        if num_sent > 3:
            z4 = zd_all[:, 2, :]  # second hard neg (optional)

        # distributed gather (same style as your current model)
        if dist.is_initialized() and self.training:
            # gather hard neg first
            if num_sent >= 3:
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                z3_list[dist.get_rank()] = z3
                z3 = torch.cat(z3_list, 0)

            zq_list = [torch.zeros_like(zq) for _ in range(dist.get_world_size())]
            zp_list = [torch.zeros_like(zp) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=zq_list, tensor=zq.contiguous())
            dist.all_gather(tensor_list=zp_list, tensor=zp.contiguous())
            zq_list[dist.get_rank()] = zq
            zp_list[dist.get_rank()] = zp
            zq = torch.cat(zq_list, 0)
            zp = torch.cat(zp_list, 0)

        loss_type = self.model_args.loss_type

        # ---------- loss: cosine InfoNCE (default) ----------
        if loss_type == "cos":
            cos_sim = self.sim(zq.unsqueeze(1), zp.unsqueeze(0))

            if num_sent >= 3:
                zq_z3_cos = self.sim(zq.unsqueeze(1), z3.unsqueeze(0))
                cos_sim = torch.cat([cos_sim, zq_z3_cos], 1)

            labels = torch.arange(cos_sim.size(0)).long().to(self.device)
            loss_fct = nn.CrossEntropyLoss()

            if num_sent == 3:
                z3_weight = self.model_args.hard_negative_weight
                weights = torch.tensor(
                    [[0.0] * (cos_sim.size(-1) - zq_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (
                                zq_z3_cos.size(-1) - i - 1)
                     for i in range(zq_z3_cos.size(-1))]
                ).to(self.device)
                cos_sim = cos_sim + weights

            loss = loss_fct(cos_sim, labels)
            self.custom_epoch_info["cl_loss"].append(loss)

        else:
            raise ValueError(f"Unknown loss_type={loss_type}")

        if not return_dict:
            output = (cos_sim,) + q_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=cos_sim,
            hidden_states=q_outputs.hidden_states,
            attentions=q_outputs.attentions,
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
            sent_emb=False,
            is_query: bool = True,  # only used when sent_emb=True
            mlm_input_ids=None,
            mlm_labels=None,
    ):
        if sent_emb:
            flag = "_dbg_printed_query" if is_query else "_dbg_printed_doc"

            encoder = self.query_bert if is_query else self.doc_bert
            return self._sentemb_forward_with(
                encoder,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        return self._dual_cl_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mlm_input_ids=mlm_input_ids,
            mlm_labels=mlm_labels,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self.model_args, "freeze_doc_encoder", True):
            self.doc_bert.eval()
        return self
