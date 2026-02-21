import logging
import os
import sys

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from arguments import ModelArguments, DataTrainingArguments, OurTrainingArguments
from collators import OurDataCollatorWithPadding
from data import load_raw_datasets, build_tokenized_datasets
from model.gte.modeling import NewModelForCL
from model.models import our_BertForCL
from model.trainers import CLTrainer

logger = logging.getLogger(__name__)


def _setup_logging(training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()


def _load_config_tokenizer(model_args: ModelArguments):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    else:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    tok_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tok_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tok_kwargs)

    return config, tokenizer


def _load_model(model_args: ModelArguments, config):
    if ("our_gte" in model_args.model_name.lower()) and (NewModelForCL is not None):
        model = NewModelForCL.from_pretrained(
            model_args.model_name_or_path,
            model_args=model_args,
            config=config,
            add_pooling_layer=True,
            trust_remote_code=True,
        )
        return model

    model = our_BertForCL.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        model_args=model_args,
    )
    return model


def _ensure_checkpoint_inputs_require_grads(model):
    """
    For gradient checkpointing:
    Make sure at least one input tensor to checkpointed blocks has requires_grad=True.
    HF provides enable_input_require_grads() which marks embedding outputs require grad,
    WITHOUT unfreezing embedding weights.
    """
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        print(">>> Enabled input require grads for gradient checkpointing (HF enable_input_require_grads)")
        return

    # Fallback for older/edge cases: hook embeddings output to require grad
    if hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
        emb = model.bert.embeddings

        def _hook(_module, _inp, out):
            if hasattr(out, "requires_grad") and (not out.requires_grad):
                out.requires_grad_(True)
            return out

        emb.register_forward_hook(_hook)
        print(">>> Enabled input require grads via embeddings forward hook (fallback)")
        return

    print(">>> [WARN] Could not enable input require grads; consider disabling --gradient_checkpointing.")


def _apply_freeze_policy(model, model_args: ModelArguments, training_args: TrainingArguments):
    """
    semantics:
      freeze_backbone = -1 : train all (no freezing)
      freeze_backbone =  0 : freeze all BERT (embeddings + encoder)
      freeze_backbone =  N : train last N encoder layers (embeddings frozen if freeze_embeddings=True)
    Always keep pooler/query_transform/lm_head trainable if they exist.
    """

    # if model has no bert, do nothing
    if not hasattr(model, "bert") or not hasattr(model.bert, "encoder") or not hasattr(model.bert.encoder, "layer"):
        return

    n_last = int(getattr(model_args, "freeze_backbone", -1))
    freeze_emb = bool(getattr(model_args, "freeze_embeddings", True))

    # -1: do nothing (full finetune)
    if n_last < 0:
        print(">>> freeze_backbone < 0: full finetune (no freezing)")
        return

    # 0 or N>0: start from freezing all BERT
    for _, p in model.bert.named_parameters():
        p.requires_grad = False

    # embeddings
    if not freeze_emb:
        for _, p in model.bert.embeddings.named_parameters():
            p.requires_grad = True

    # encoder layers
    if n_last == 0:
        pass
    else:
        layers = list(model.bert.encoder.layer)
        total = len(layers)
        n_unfreeze = min(n_last, total)
        for i in range(total - n_unfreeze, total):
            for _, p in layers[i].named_parameters():
                p.requires_grad = True

    # always allow these heads to train (if exist)
    for n, p in model.named_parameters():
        if ("pooler" in n) or ("query_transform" in n) or ("lm_head" in n):
            p.requires_grad = True

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(">>> Applied freeze_backbone policy")
    print(f">>> freeze_backbone={n_last} (train last N layers), freeze_embeddings={freeze_emb}")
    print(f">>> #trainable={len(trainable)}")
    for n in trainable[:80]:
        print("   ", n)
    if len(trainable) > 80:
        print(f"   ... ({len(trainable) - 80} more)")

    if len(trainable) == 0:
        raise ValueError(
            "After applying freeze policy, no parameters are trainable. "
            "Use --freeze_backbone -1 or >0, or enable a trainable head."
        )

    # ---- CRITICAL FIX ----
    # If using gradient checkpointing, ensure checkpoint inputs require grads.
    if bool(getattr(training_args, "gradient_checkpointing", False)):
        _ensure_checkpoint_inputs_require_grads(model)


def main():
    os.environ["WANDB_DISABLED"] = "true"

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) exists and is not empty. "
            "Use --overwrite_output_dir to overwrite."
        )

    _setup_logging(training_args)
    logger.warning(
        f"rank={training_args.local_rank}, device={training_args.device}, "
        f"distributed={training_args.local_rank != -1}, fp16={training_args.fp16}"
    )
    set_seed(training_args.seed)

    config, tokenizer = _load_config_tokenizer(model_args)
    if training_args.gradient_checkpointing:
        config.gradient_checkpointing = True
        # recommended for checkpointing in transformers models
        if hasattr(config, "use_cache"):
            config.use_cache = False

    model = _load_model(model_args, config)
    model.resize_token_embeddings(len(tokenizer))

    # apply freeze policy BEFORE trainer/optimizer
    _apply_freeze_policy(model, model_args, training_args)

    raw = load_raw_datasets(
        train_file=data_args.train_file,
        eval_file=data_args.eval_file if training_args.do_eval else None,
        cache_dir="./data/",
    )
    tokenized = build_tokenized_datasets(
        raw_datasets=raw,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        pad_to_max_length=data_args.pad_to_max_length,
        preprocessing_num_workers=data_args.preprocessing_num_workers,
        overwrite_cache=data_args.overwrite_cache,
    )

    train_dataset = tokenized["train"] if training_args.do_train else None
    eval_dataset = tokenized.get("validation", None) if training_args.do_eval else None

    data_collator = OurDataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        do_mlm=bool(model_args.do_mlm),
        mlm_probability=float(data_args.mlm_probability),
    )

    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        model_args=model_args,
    )

    if training_args.do_train:
        resume = model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        train_result = trainer.train(resume_from_checkpoint=resume)
        trainer.save_model()
        if trainer.is_world_process_zero():
            with open(os.path.join(training_args.output_dir, "train_results.txt"), "w", encoding="utf-8") as f:
                for k, v in sorted(train_result.metrics.items()):
                    f.write(f"{k} = {v}\n")
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    if training_args.do_eval:
        metrics = trainer.evaluate()
        if trainer.is_world_process_zero():
            with open(os.path.join(training_args.output_dir, "eval_results.txt"), "w", encoding="utf-8") as f:
                for k, v in sorted(metrics.items()):
                    f.write(f"{k} = {v}\n")


if __name__ == "__main__":
    main()
