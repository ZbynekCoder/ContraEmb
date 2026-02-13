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

    # Freeze
    freeze_backbone: bool = field(default=False)

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
