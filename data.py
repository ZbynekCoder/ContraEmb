from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def load_raw_datasets(
    train_file: str,
    eval_file: Optional[str] = None,
    cache_dir: str = "./data/",
):
    """
    Load dataset from local files via datasets.load_dataset.

    Matches original alignment logic for CSV/TSV:
      - Read TRAIN header
      - Build datasets.Features from TRAIN header
      - load_dataset(..., features=features) for train+validation
    No extra "补列兜底". If schema mismatches, it should error early.
    """
    if not train_file:
        raise ValueError("train_file is required.")

    data_files = {"train": train_file}
    if eval_file:
        data_files["validation"] = eval_file

    ext = train_file.split(".")[-1].lower()

    if ext == "txt":
        return load_dataset("text", data_files=data_files, cache_dir=cache_dir)

    if ext in ("csv", "tsv"):
        from datasets import Features, Value
        import csv as pycsv

        delimiter = "\t" if ext == "tsv" else ","
        with open(train_file, newline="", encoding="utf-8") as f:
            reader = pycsv.reader(f, delimiter=delimiter)
            header = next(reader)

        features = Features({col: Value("string") for col in header})

        return load_dataset(
            "csv",
            data_files=data_files,
            delimiter=delimiter,
            features=features,
            cache_dir=cache_dir,
        )

    if ext == "json":
        return load_dataset("json", data_files=data_files, cache_dir=cache_dir)

    raise ValueError(f"Unsupported train_file extension: {ext}. Use csv/tsv/json/txt.")


def infer_columns(column_names: List[str]) -> Tuple[str, str, List[str]]:
    """
    Infer (sent0, sent1, hard_cols) from TRAIN columns.
    """
    if len(column_names) == 1:
        c0 = column_names[0]
        return c0, c0, []
    if len(column_names) >= 2:
        return column_names[0], column_names[1], column_names[2:]
    raise ValueError("Dataset must have at least 1 column.")


def build_prepare_features(
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    pad_to_max_length: bool,
    sent0_cname: str,
    sent1_cname: str,
    hard_cnames: List[str],
):
    """
    Return a function(examples) -> tokenized features:
      - Each example becomes [q, pos, hard1, hard2, ...] (variable length)
      - Output keys map to nested lists: feature[key][i] = [sent0_ids, sent1_ids, ...]
      - Also outputs 'num_hard'
    """

    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return [t if t is not None else " " for t in x if t is not None]
        return [x]

    def prepare_features(examples: Dict[str, List]):
        total = len(examples[sent0_cname])

        flat_sentences: List[str] = []
        sent_counts: List[int] = []
        num_hard_list: List[int] = []

        for i in range(total):
            q = examples[sent0_cname][i] if examples[sent0_cname][i] is not None else " "
            p = examples[sent1_cname][i] if examples[sent1_cname][i] is not None else " "

            hard_list: List[str] = []
            # IMPORTANT: no schema fallback. If examples[hc] missing -> KeyError (desired)
            for hc in hard_cnames:
                hard_list.extend(_as_list(examples[hc][i]))

            hard_list = [h for h in hard_list if h is not None and len(str(h).strip()) > 0]

            flat_sentences.append(q)
            flat_sentences.append(p)
            flat_sentences.extend(hard_list)

            num_hard = len(hard_list)
            num_hard_list.append(num_hard)
            sent_counts.append(2 + num_hard)

        tokenized = tokenizer(
            flat_sentences,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length" if pad_to_max_length else False,
        )

        # regroup to per-example list-of-sentences
        features: Dict[str, List[List[List[int]]]] = {k: [] for k in tokenized.keys()}
        offset = 0
        for i in range(total):
            n = sent_counts[i]
            for k in tokenized.keys():
                features[k].append([tokenized[k][offset + j] for j in range(n)])
            offset += n

        features["num_hard"] = num_hard_list
        return features

    return prepare_features


def build_tokenized_datasets(
    raw_datasets,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    pad_to_max_length: bool,
    preprocessing_num_workers: Optional[int],
    overwrite_cache: bool,
):
    """
    Map raw datasets into tokenized datasets with nested sentence lists.
    No validation column padding. If train/val schema mismatches, it should error.
    """
    train_columns = raw_datasets["train"].column_names
    sent0_cname, sent1_cname, hard_cnames = infer_columns(train_columns)

    prepare_features = build_prepare_features(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        pad_to_max_length=pad_to_max_length,
        sent0_cname=sent0_cname,
        sent1_cname=sent1_cname,
        hard_cnames=hard_cnames,
    )

    tokenized = {}
    tokenized["train"] = raw_datasets["train"].map(
        prepare_features,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=train_columns,
        load_from_cache_file=not overwrite_cache,
    )

    if "validation" in raw_datasets:
        val_columns = raw_datasets["validation"].column_names
        tokenized["validation"] = raw_datasets["validation"].map(
            prepare_features,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=val_columns,
            load_from_cache_file=not overwrite_cache,
        )

    return tokenized
