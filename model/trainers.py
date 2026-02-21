import os
from typing import List, Tuple

import torch
from torch.optim import AdamW
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback


class CustomInfoCallback(TrainerCallback):
    def __init__(self, every_n_steps: int = 0, filename: str = "custom_info"):
        self.every_n_steps = int(every_n_steps)
        self.filename = filename

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_n_steps <= 0:
            return control
        if state.global_step <= 0 or (state.global_step % self.every_n_steps != 0):
            return control

        model = kwargs.get("model", None)
        if model is None:
            return control

        actual = model.module if hasattr(model, "module") else model
        if not hasattr(actual, "custom_epoch_info"):
            return control

        path = os.path.join(args.output_dir, self.filename)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"step={state.global_step} info={actual.custom_epoch_info}\n")
        return control


def _split_decay_params(named_params: List[Tuple[str, torch.nn.Parameter]]):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay_params = []
    no_decay_params = []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            no_decay_params.append((n, p))
        else:
            decay_params.append((n, p))
    return decay_params, no_decay_params


class CLTrainer(Trainer):
    """
    HF Trainer + optional callback + optimizer param groups:
      - BERT params use bert_lr (or base_lr * bert_lr_scale)
      - query_transform uses query_transform_lr (or base_lr)
      - other heads default to bert_lr (stable)
    """

    def __init__(
        self,
        *args,
        model_args=None,
        custom_info_every_n_steps: int = 0,
        custom_info_filename: str = "custom_info",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_args = model_args

        if int(custom_info_every_n_steps) > 0:
            self.add_callback(
                CustomInfoCallback(
                    every_n_steps=int(custom_info_every_n_steps),
                    filename=custom_info_filename,
                )
            )

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        # if no model_args -> fallback
        if self.model_args is None:
            return super().create_optimizer()

        model = self.model.module if hasattr(self.model, "module") else self.model

        # sanity: must have at least one trainable parameter
        trainable_cnt = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        if trainable_cnt == 0:
            raise ValueError(
                "No trainable parameters found (all requires_grad=False). "
                "Check --freeze_backbone / --freeze_embeddings / --use_query_transform / --do_mlm."
            )

        base_lr = float(self.args.learning_rate)
        qt_lr = float(self.model_args.query_transform_lr) if getattr(self.model_args, "query_transform_lr", None) is not None else base_lr
        bert_lr = float(self.model_args.bert_lr) if getattr(self.model_args, "bert_lr", None) is not None else base_lr * float(getattr(self.model_args, "bert_lr_scale", 0.1))

        qt_named = []
        bert_named = []
        other_named = []

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "query_transform" in n:
                qt_named.append((n, p))
            elif n.startswith("bert.") or ".bert." in n:
                bert_named.append((n, p))
            else:
                other_named.append((n, p))

        groups = []

        def add_groups(named_params: List[Tuple[str, torch.nn.Parameter]], lr: float):
            decay, no_decay = _split_decay_params(named_params)
            if decay:
                groups.append({"params": [p for _, p in decay], "weight_decay": float(self.args.weight_decay), "lr": float(lr)})
            if no_decay:
                groups.append({"params": [p for _, p in no_decay], "weight_decay": 0.0, "lr": float(lr)})

        add_groups(bert_named, bert_lr)
        add_groups(qt_named, qt_lr)
        add_groups(other_named, bert_lr)

        optim_kwargs = {
            "lr": base_lr,  # overridden by groups
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
        }
        self.optimizer = AdamW(groups, **optim_kwargs)
        return self.optimizer

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        losses = []
        with torch.no_grad():
            for batch in dataloader:
                out = self.model(**batch, output_hidden_states=True, return_dict=True, sent_emb=False)
                loss = out.loss
                if loss is not None:
                    losses.append(loss.detach().float())

        avg = torch.stack(losses).mean().item() if losses else 0.0
        return {f"{metric_key_prefix}_loss": avg}
