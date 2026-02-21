from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Pretrained checkpoint path"})
    tokenizer_name: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)

    # CL
    temp: float = field(default=0.05)
    pooler_type: str = field(default="cls")

    # Query transform
    use_query_transform: bool = field(default=False)
    query_transform_dropout: float = field(default=0.1)
    query_transform_scale: float = field(default=1.0)
    query_transform_init_std: float = field(default=0.02)
    query_transform_type: str = field(default="gated_mlp")
    query_transform_mlp_ratio: float = field(default=0.25)

    # Loss
    loss_type: str = field(default="cos")  # cos | decouple
    tau_E: float = field(default=None)
    stance_margin: float = field(default=0.1)
    stance_beta: float = field(default=10.0)
    stance_alpha: float = field(default=1.0)

    # MLM
    do_mlm: bool = field(default=False)
    mlm_weight: float = field(default=0.1)

    # -------------------------
    # Freeze policy (NEW semantics)
    # -------------------------
    # -1: full finetune (no freezing)
    #  0: freeze all BERT (embeddings + encoder)
    # >0: unfreeze last N encoder layers (default embeddings frozen unless freeze_embeddings=False)
    freeze_backbone: int = field(
        default=-1,
        metadata={"help": "How many last BERT encoder layers to train. -1=all, 0=freeze all, N>0=train last N."},
    )

    # Freeze embeddings (word/position/token-type + emb LayerNorm)
    freeze_embeddings: bool = field(
        default=True,
        metadata={"help": "Freeze BERT embeddings when freeze_backbone>=0. Default True."},
    )

    # LR strategy (optional, keep for future)
    query_transform_lr: Optional[float] = field(
        default=None,
        metadata={"help": "LR for query_transform. Default: same as TrainingArguments.learning_rate."},
    )
    bert_lr: Optional[float] = field(
        default=None,
        metadata={"help": "LR for BERT params. Default: base_lr * bert_lr_scale."},
    )
    bert_lr_scale: float = field(
        default=0.1,
        metadata={"help": "If bert_lr is None, use base_lr * bert_lr_scale."},
    )

    # Model switch (kept for compatibility)
    model_name: str = field(default="bge")  # e.g. our_gte/...


@dataclass
class DataTrainingArguments:
    train_file: str = field(metadata={"help": "Training data file (csv/tsv/json/txt)"})
    eval_file: Optional[str] = field(default=None)

    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)

    max_seq_length: int = field(default=32)
    pad_to_max_length: bool = field(default=False)
    mlm_probability: float = field(default=0.15)


@dataclass
class OurTrainingArguments(TrainingArguments):
    eval_transfer: bool = field(default=False)
