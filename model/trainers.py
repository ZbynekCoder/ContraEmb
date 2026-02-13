import os
import torch
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback


class CustomInfoCallback(TrainerCallback):
    """
    Optional: periodically dump model.custom_epoch_info to a file.
    Enabled only when `every_n_steps > 0`.
    """
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
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"step={state.global_step} info={actual.custom_epoch_info}\n")
        except Exception:
            # avoid crashing training due to logging
            pass

        return control


class CLTrainer(Trainer):
    """
    Minimal CLTrainer:
    - does NOT override train(): uses HF standard training loop
    - optional: log custom_epoch_info via callback (disabled by default)
    - evaluate(): returns avg eval loss as {'eval_loss': float}
    """

    def __init__(
        self,
        *args,
        custom_info_every_n_steps: int = 0,
        custom_info_filename: str = "custom_info",
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Drop-in: no need to change train.py, just pass custom_info_every_n_steps if you want.
        if int(custom_info_every_n_steps) > 0:
            self.add_callback(
                CustomInfoCallback(
                    every_n_steps=int(custom_info_every_n_steps),
                    filename=custom_info_filename,
                )
            )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Return avg loss over eval dataloader.
        This matches HF expectations (a dict of floats), enabling:
        - load_best_model_at_end
        - metric_for_best_model="loss" or "eval_loss"
        """
        dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        losses = []
        with torch.no_grad():
            for batch in dataloader:
                out = self.model(**batch, output_hidden_states=True, return_dict=True, sent_emb=False)
                loss = out.loss
                if loss is not None:
                    losses.append(loss.detach().float())

        if not losses:
            avg = 0.0
        else:
            avg = torch.stack(losses).mean().item()

        return {f"{metric_key_prefix}_loss": avg}
