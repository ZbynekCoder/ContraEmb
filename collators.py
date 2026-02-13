from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase


@dataclass
class OurDataCollatorWithPadding:
    """
    Collator producing:
      - input_ids/attention_mask/token_type_ids: (bs, max_num_sent, L)
      - hard_mask: (bs, max_hard) bool, valid hard negatives for positions 2..
      - optional mlm_input_ids/mlm_labels (same shape as input_ids) when do_mlm=True
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    # MLM
    do_mlm: bool = False
    mlm_probability: float = 0.15

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        bs = len(features)
        if bs == 0:
            return {}

        # each feature has: input_ids = [sent0_ids, sent1_ids, ...]
        num_sent_list = [len(f["input_ids"]) for f in features]
        max_num_sent = max(num_sent_list)
        max_hard = max(0, max_num_sent - 2)

        flat_features: List[Dict] = []
        hard_mask = torch.zeros((bs, max_hard), dtype=torch.bool)

        for b, feat in enumerate(features):
            n_sent = len(feat["input_ids"])
            n_hard = max(0, n_sent - 2)
            if n_hard > 0 and max_hard > 0:
                hard_mask[b, :n_hard] = True

            # push real sentences
            for i in range(n_sent):
                one = {
                    "input_ids": feat["input_ids"][i],
                    "attention_mask": feat["attention_mask"][i],
                }
                if "token_type_ids" in feat:
                    one["token_type_ids"] = feat["token_type_ids"][i]
                flat_features.append(one)

            # pad missing sentences with dummy (will be padded by tokenizer.pad)
            pad_needed = max_num_sent - n_sent
            if pad_needed > 0:
                dummy = {
                    "input_ids": [self.tokenizer.pad_token_id],
                    "attention_mask": [0],
                }
                if "token_type_ids" in feat:
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

        if self.do_mlm:
            batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

        # reshape back: (bs, max_num_sent, L)
        for k in list(batch.keys()):
            if k in ["input_ids", "attention_mask", "token_type_ids", "mlm_input_ids", "mlm_labels"]:
                batch[k] = batch[k].view(bs, max_num_sent, -1)

        batch["hard_mask"] = hard_mask

        # normalize label keys if exist
        if "label" in batch:
            batch["labels"] = batch.pop("label")
        if "label_ids" in batch:
            batch["labels"] = batch.pop("label_ids")

        return batch

    def mask_tokens(
        self,
        inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard BERT MLM masking:
          - 80% [MASK], 10% random, 10% unchanged among masked positions
        """
        if self.tokenizer.mask_token is None:
            raise ValueError("This tokenizer does not have a mask token for MLM.")

        inputs = inputs.clone()
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, float(self.mlm_probability))
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% -> [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% -> random word (of the remaining 20%)
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # 10% -> unchanged: the remaining masked positions
        return inputs, labels
