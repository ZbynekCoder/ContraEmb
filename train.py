import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import collections
import random

from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
# from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from transformers.file_utils import cached_property, is_torch_available, is_torch_tpu_available
from model.models import our_BertForCL
from model.trainers import CLTrainer

from model.gte.modeling import NewModelForCL

from torch.utils.data import random_split

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )

    # our's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    )

    # ====== ADD THIS INTO ModelArguments (train.py) ======
    use_query_transform: bool = field(
        default=False,
        metadata={"help": "If True, apply a learnable transform T on anchor/query embeddings only: z1 <- T(z1)."}
    )
    query_transform_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout used inside query transform T."}
    )
    query_transform_scale: float = field(
        default=1.0,
        metadata={"help": "Scale for residual transform: T(z)=normalize(z + scale*Dropout(Wz))."}
    )
    query_transform_init_std: float = field(
        default=0.02,
        metadata={"help": "Init std for query_transform weights (normal distribution)."}
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={
            "help": "If True, freeze the encoder backbone and only train query-side modules (e.g., query_transform)."}
    )
    query_transform_type: str = field(
        default="gated_mlp",
        metadata={"help": "Query transform type: linear | mlp | gated_mlp"}
    )
    query_transform_mlp_ratio: float = field(
        default=0.25,
        metadata={"help": "Hidden ratio for query transform MLP (e.g., 0.25 / 0.5 / 1.0)"}
    )
    # ====== END ADD ======

    # ====== Route C: asymmetric dual-encoder ======
    use_dual_encoder: bool = field(
        default=False,
        metadata={"help": "If True, use DualBertForCL: doc tower frozen + query tower finetune."}
    )
    doc_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional doc-tower checkpoint. If None, use model_name_or_path."}
    )
    freeze_doc_encoder: bool = field(
        default=True,
        metadata={"help": "If True, freeze doc tower (Route C default)."}
    )

    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )
    model_name: str = field(
        default="bge",
        metadata={
            "help": "which type of model you are using"
        },
    )
    loss_type: str = field(
        default="cos",
        metadata={
            "help": "which loss function to use"
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments.
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "The validation data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    @cached_property
    # @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        self.distributed_state = None

        return device


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    os.environ["WANDB_DISABLED"] = "true"

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("save steps", training_args.save_steps)

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.eval_file is not None:
        data_files["validation"] = data_args.eval_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        # datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",")
        from datasets import Features, Value
        import csv

        train_path = data_files["train"] if isinstance(data_files, dict) else data_files
        with open(train_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t" if "tsv" in data_args.train_file else ",")
            header = next(reader)

        features = Features({col: Value("string") for col in header})

        datasets = load_dataset(
            extension,
            data_files=data_files,
            delimiter="\t" if "tsv" in data_args.train_file else ",",
            features=features,
        )

    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")

    # print(datasets)
    # datasets["train"] = datasets["train"].select(range(1000))

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs, trust_remote_code=True)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs, trust_remote_code=True)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    if training_args.gradient_checkpointing:
        config.gradient_checkpointing = True

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            **tokenizer_kwargs,
            trust_remote_code=True)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            **tokenizer_kwargs,
            trust_remote_code=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        if "our_bge" in model_args.model_name or "our_uae" in model_args.model_name:
            model = our_BertForCL.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                    model_args=model_args
                )
        elif "our_gte" in model_args.model_name:
            model = NewModelForCL.from_pretrained(
                    model_args.model_name_or_path,
                    model_args=model_args,
                    config=config,
                    add_pooling_layer=True,
                    trust_remote_code=True,
                )
    else:
        raise NotImplementedError

    new_n = len(tokenizer)
    old_n = getattr(model.config, "vocab_size", None)

    if old_n is None or new_n != old_n:
        model.resize_token_embeddings(new_n)

    # ================== FREEZE POLICY ==================
    if model_args.freeze_backbone:
        print(">>> Freezing backbone encoder parameters")
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "query_transform" in name:
                param.requires_grad = True
            if "pooler" in name:
                param.requires_grad = True
        

    # sanity check：print trainables
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(">>> Trainable parameters:")
    for n in trainable:
        print("   ", n)
    # ================== END FREEZE ==================

    # Prepare features
    column_names = datasets["train"].column_names

    # Expect at least two columns: anchor/query and positive.
    # Additional columns (if any) are treated as hard negatives.
    # Hard negatives can be provided either as:
    #   (a) one or more string columns (each column is one hard negative), or
    #   (b) a single list-of-strings column (variable number of hard negatives per example).
    if len(column_names) == 1:
        # Unsupervised datasets: use same column twice
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
        hard_cnames: List[str] = []
    elif len(column_names) >= 2:
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        hard_cnames = column_names[2:]
    else:
        raise NotImplementedError

    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            # filter None
            return [t if t is not None else " " for t in x if t is not None]
        # scalar (string)
        return [x]

    def prepare_features(examples):
        total = len(examples[sent0_cname])

        flat_sentences: List[str] = []
        sent_counts: List[int] = []  # per-example number of sentences = 2 + num_hard
        num_hard_list: List[int] = []

        # Build flattened sentence list: [q, d+, hard...]
        for idx in range(total):
            q_text = examples[sent0_cname][idx] if examples[sent0_cname][idx] is not None else " "
            p_text = examples[sent1_cname][idx] if examples[sent1_cname][idx] is not None else " "

            hard_list: List[str] = []
            for hc in hard_cnames:
                hard_list.extend(_as_list(examples[hc][idx]))

            hard_list = [
                h for h in hard_list
                if h is not None and len(str(h).strip()) > 0
            ]

            flat_sentences.append(q_text)
            flat_sentences.append(p_text)
            for h in hard_list:
                flat_sentences.append(h)

            num_hard = len(hard_list)
            num_hard_list.append(num_hard)
            sent_counts.append(2 + num_hard)

        sent_features = tokenizer(
            flat_sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Re-pack back to per-example variable-length list of sentences
        features: Dict[str, List[List[List[int]]]] = {}
        offset = 0
        for key in sent_features:
            features[key] = []
        for idx in range(total):
            n_sent = sent_counts[idx]
            for key in sent_features:
                features[key].append([sent_features[key][offset + j] for j in range(n_sent)])
            offset += n_sent

        # Store num_hard for downstream collator / loss masking
        features["num_hard"] = num_hard_list
        return features

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    if training_args.do_eval:
        validation_dataset = datasets["validation"].map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        validation_size = len(validation_dataset)
        validation_dataset = \
            random_split(validation_dataset, [validation_size], generator=torch.Generator().manual_seed(42))[0]

    # Data collator
    @dataclass
    class OurDataCollatorWithPadding:

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        mlm: bool = True
        mlm_probability: float = data_args.mlm_probability

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[
            str, torch.Tensor]:
            # Each feature contains variable-length list of sentences:
            #   feature["input_ids"] = [q, d+, hard1, hard2, ...]
            # We pad the number of sentences in the batch to max_num_sent,
            # and create a hard_mask for valid hard negatives (positions >=2).
            special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
            bs = len(features)
            if bs == 0:
                return {}

            num_sent_list = [len(f['input_ids']) for f in features]
            max_num_sent = max(num_sent_list)
            max_hard = max(0, max_num_sent - 2)

            flat_features = []
            hard_mask = torch.zeros((bs, max_hard), dtype=torch.bool)
            for b, feature in enumerate(features):
                n_sent = len(feature['input_ids'])
                n_hard = max(0, n_sent - 2)

                # mark valid hard negatives for this example
                if n_hard > 0 and max_hard > 0:
                    hard_mask[b, :n_hard] = True

                # add existing sentences
                for i in range(n_sent):
                    flat_features.append({k: feature[k][i] if k in special_keys else feature.get(k) for k in feature})

                # pad missing sentences with a minimal dummy sequence; tokenizer.pad will expand it
                pad_needed = max_num_sent - n_sent
                if pad_needed > 0:
                    dummy = {
                        "input_ids": [self.tokenizer.pad_token_id],
                        "attention_mask": [0],
                    }
                    if "token_type_ids" in feature:
                        dummy["token_type_ids"] = [0]
                    for _ in range(pad_needed):
                        flat_features.append(dummy)

            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            # MLM auxiliary objective (optional)
            if model_args.do_mlm:
                batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

            # reshape back: (bs, max_num_sent, L)
            for k in list(batch.keys()):
                if k in special_keys:
                    batch[k] = batch[k].view(bs, max_num_sent, -1)

            # add hard_mask to batch for loss masking
            batch["hard_mask"] = hard_mask

            # normalize label field names if present
            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]

            return batch

        def mask_tokens(
                self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Prepare masked tokens inputs/labels for masked language modeling."""
            inputs = inputs.clone()
            labels = inputs.clone()
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                    labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100

            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            return inputs, labels

    # data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer)
    data_collator = OurDataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.model_args = model_args

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        pass

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
