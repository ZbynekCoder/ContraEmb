import logging
import math
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions, \
    BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead, BertLayer, \
    BertEmbeddings, BertPooler

logger = logging.getLogger(__name__)


class OurBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = config.gradient_checkpointing if hasattr(config,
                                                                               'gradient_checkpointing') else False

    def _gradient_checkpointing_func(self, layer_call, *args, **kwargs):
        tensor_args = tuple(arg for arg in args if isinstance(arg, torch.Tensor))
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
        self.post_init()


class MLPLayer(nn.Module):
    def __init__(self, config, output_dim=None):
        super().__init__()
        out_dim = output_dim if output_dim is not None else config.hidden_size
        self.dense = nn.Linear(config.hidden_size, out_dim)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2",
                                    "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
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
    初始化双塔 Head 和相似度模块
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)

    # 获取维度配置
    content_dim = getattr(cls.model_args, 'content_head_dim', config.hidden_size)
    stance_dim = getattr(cls.model_args, 'stance_head_dim', 128)

    # 1. 初始化两个独立的 Head
    cls.content_head = MLPLayer(config, output_dim=content_dim)
    cls.stance_head = MLPLayer(config, output_dim=stance_dim)

    # 作用：拟合流形上的非线性截面，将 content 向量映射回 hidden_state 的局部主成分
    cls.content_removal_proj = nn.Sequential(
        nn.Linear(content_dim, config.hidden_size),
        nn.LayerNorm(config.hidden_size),  # 归一化防止数值爆炸
        nn.GELU(),  # 引入非线性，适应弯曲流形
        nn.Linear(config.hidden_size, config.hidden_size)
    )

    # 初始化策略
    with torch.no_grad():
        for m in cls.content_removal_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

    cls.hidden_size = config.hidden_size
    cls.sqrt_hidden_size = math.sqrt(cls.hidden_size)

    cls.current_training_progress = 0
    print(f"Model Init: Hidden={cls.hidden_size}, Proj=MLP(Non-linear)")


def our_cl_forward(cls, encoder, input_ids, attention_mask, token_type_ids, phase=1,
                   qr_loss_weight=0.0, stance_loss_weight=0.0, content_loss_weight=1.0, **kwargs):
    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1)  # 应为 3: [Q, Contra, Sim]

    # Flatten & Encode
    outputs = encoder(input_ids.view(-1, input_ids.size(-1)),
                      attention_mask=attention_mask.view(-1, attention_mask.size(-1)),
                      token_type_ids=token_type_ids.view(-1, token_type_ids.size(
                          -1)) if token_type_ids is not None else None,
                      return_dict=True)

    h_pooled = cls.pooler(attention_mask.view(-1, attention_mask.size(-1)), outputs)

    # 1. 获取内容向量
    all_a = cls.content_head(h_pooled)

    # 2. 投影
    a_projected = cls.content_removal_proj(all_a)

    # 3. 逆残差
    b_input = h_pooled - a_projected
    all_b = cls.stance_head(b_input)

    # 按照 [bs, 3, dim] 解包
    a_z = all_a.view(batch_size, num_sent, -1)
    b_z = all_b.view(batch_size, num_sent, -1)

    # 物理意义对齐: 0=Q, 1=Contra, 2=Sim
    a_q, a_contra, a_sim = a_z[:, 0], a_z[:, 1], a_z[:, 2]
    b_q, b_contra, b_sim = b_z[:, 0], b_z[:, 1], b_z[:, 2]

    total_loss = torch.tensor(0.0, device=cls.device, requires_grad=True)
    loss_fct = nn.CrossEntropyLoss()

    # --- Phase 1: 内容核提纯 ---
    if phase == 1 and content_loss_weight > 0:
        sim_q_contra = cls.sim(a_q.unsqueeze(1), a_contra.unsqueeze(0))
        sim_q_sim = cls.sim(a_q.unsqueeze(1), a_sim.unsqueeze(0))
        logits = torch.cat([sim_q_contra, sim_q_sim], dim=1)
        labels = torch.arange(batch_size).long().to(cls.device)
        loss_c = (loss_fct(logits, labels) + loss_fct(logits, labels + batch_size)) / 2
        total_loss = total_loss + content_loss_weight * loss_c
        cls.custom_epoch_info["content_loss_log"].append(loss_c.detach().cpu().item())

    # --- Phase 2: 立场流形塑造 ---
    elif phase == 2:
        if stance_loss_weight > 0:
            sim_q_sim = cls.sim(b_q.unsqueeze(1), b_sim.unsqueeze(0))
            sim_q_contra = cls.sim(b_q.unsqueeze(1), b_contra.unsqueeze(0))
            logits_s = torch.cat([sim_q_sim, sim_q_contra], dim=1)
            labels_s = torch.arange(batch_size).long().to(cls.device)
            loss_s = loss_fct(logits_s, labels_s)
            total_loss = total_loss + stance_loss_weight * loss_s
            cls.custom_epoch_info["stance_loss_log"].append(loss_s.detach().cpu().item())

        if qr_loss_weight > 0:
            # 计算每个样本内部 a 和 b 的余弦相似度
            # dim=-1 表示沿着向量维度计算
            cos_sim = torch.nn.functional.cosine_similarity(a_projected, b_input, dim=-1)

            # 目标是垂直 (Cosine=0)，所以最小化 Cosine 的平方
            # 这比绝对值梯度更平滑，且对大偏差惩罚更重
            local_orth_loss = torch.mean(cos_sim ** 2)

            total_loss = total_loss + qr_loss_weight * local_orth_loss
            # 依然记录为 qr_loss_log 以兼容旧脚本
            cls.custom_epoch_info["qr_loss_log"].append(local_orth_loss.detach().cpu().item())

    cls.custom_epoch_info["total_loss_log"].append(total_loss.detach().cpu().item())
    return SequenceClassifierOutput(loss=total_loss, hidden_states=outputs.hidden_states)


def sentemb_forward(
        cls,
        encoder,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        return_dict=None,
        **kwargs
):
    """
    推理模式：必须与训练逻辑保持高度一致！
    """
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    outputs = encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
    h_pooled = cls.pooler(attention_mask, outputs)

    # 1. 提取内容
    content_vec = cls.content_head(h_pooled)

    # 2. 非线性投影 (Local Manifold Estimation)
    a_projected = cls.content_removal_proj(content_vec)

    # 3. 逆残差 (Orthogonal Subtraction)
    b_input = h_pooled - a_projected

    # 4. 提取立场
    stance_vec = cls.stance_head(b_input)

    return {"content": content_vec, "stance": stance_vec}


class our_BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs.get("model_args", None)
        if self.model_args is None: self.model_args = config

        self.bert = BertModel(config, add_pooling_layer=False)

        if hasattr(self.model_args, 'do_mlm') and self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        self.custom_epoch_info = {
            "content_loss_log": [],
            "stance_loss_log": [],
            "qr_loss_log": [],
            "total_loss_log": []
        }
        cl_init(self, config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,
                sent_emb=False, mlm_input_ids=None, mlm_labels=None,
                phase=1, qr_loss_weight=0.0, stance_loss_weight=0.0, content_loss_weight=1.0):
        if sent_emb:
            return sentemb_forward(self, self.bert, input_ids=input_ids, attention_mask=attention_mask,
                                   token_type_ids=token_type_ids, return_dict=return_dict)
        else:
            return our_cl_forward(self, self.bert, input_ids=input_ids, attention_mask=attention_mask,
                                  token_type_ids=token_type_ids, phase=phase,
                                  qr_loss_weight=qr_loss_weight, stance_loss_weight=stance_loss_weight,
                                  content_loss_weight=content_loss_weight)

    # --- Save/Load Custom Heads ---
    def _save_pretrained(self, save_directory, state_dict=None, **kwargs):
        if state_dict is None: state_dict = self.state_dict()

        content_keys = [k for k in state_dict.keys() if
                        k.startswith("content_head.") or k.startswith("content_removal_proj.")]
        stance_keys = [k for k in state_dict.keys() if k.startswith("stance_head.")]
        torch.save({k: state_dict[k] for k in content_keys}, f"{save_directory}/content_head.bin")
        torch.save({k: state_dict[k] for k in stance_keys}, f"{save_directory}/stance_head.bin")
        rest_keys = [k for k in state_dict.keys() if k not in content_keys and k not in stance_keys]
        super()._save_pretrained(save_directory, state_dict={k: state_dict[k] for k in rest_keys}, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model
