#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case study script for contradiction retrieval:
Compare:
  (A) Fine-tuned model with query-transform scoring: <T(E_ft(q)), E_ft(d)>
  (B) Original (zero-shot) embedding model scoring: <E_base(q), E_base(d)>

This is meant for qualitative / ablation case study (top-K lists), not for full evaluation only.

It follows the same data pickle convention as test.py:
  ./data/arguana_{train|dev|test}_retrieval_final.pkl
  ./data/arguana_{split}_retrieval_final.pkl  (contains: ignore, queries, qrels)
and for synthetic datasets:
  ./data/{dataset}_{train|dev|test}_retrieval_gpt4_final.pkl
  ./data/{dataset}_{split}_retrieval_gpt4_final.pkl
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"

import json
import pickle
import random
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig, HfArgumentParser

# Needed to load your fine-tuned checkpoint (same as test.py)
from model.models import our_BertForCL
try:
    from model.gte.modeling import NewModelForCL
except Exception:
    NewModelForCL = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -------------------------
# Arguments
# -------------------------
@dataclass
class ModelArguments:
    # Fine-tuned checkpoint (ours)
    finetuned_model_name_or_path: str = field(
        metadata={"help": "Path to a fine-tuned checkpoint directory (ours)."}
    )
    # Baseline embedding model (original)
    baseline_model_name: str = field(
        default="BAAI/bge-base-en-v1.5",
        metadata={"help": "HF model name for baseline encoder (original)."}
    )

    # embedding settings
    pooler_type: str = field(default="avg")
    max_seq_length: int = field(default=512)
    batch_size: int = field(default=64)
    use_data_parallel: bool = field(default=False)

    # query-side transform toggle used inside our_* models
    use_query_transform: bool = field(default=True)
    # keep these for compatibility with your model init
    query_transform_scale: float = field(default=1.0)
    query_transform_dropout: float = field(default=0.1)
    query_transform_init_std: float = field(default=0.02)
    query_transform_type: str = field(default="gated_mlp")
    query_transform_mlp_ratio: float = field(default=0.25)

    # compatibility fields
    do_mlm: bool = field(default=False)
    mlm_weight: float = field(default=0.1)
    mlp_only_train: bool = field(default=False)
    temp: float = field(default=0.02)
    hard_negative_weight: float = field(default=0.0)
    loss_type: str = field(default="cos")


@dataclass
class DataArguments:
    dataset_name: str = field(default="arguana")
    split: str = field(default="test", metadata={"help": "dev | test"})
    data_dir: str = field(default="./data")


@dataclass
class CaseArguments:
    write_path: str = field(default="case_results")
    index_dir: str = field(default="./indices_case")
    overwrite_index: bool = field(default=False)

    # retrieval / case config
    k_neighbors: int = field(default=1000, metadata={"help": "FAISS search depth."})
    case_topk: int = field(default=10, metadata={"help": "Top-K to dump for each query."})
    case_max_q: int = field(default=30, metadata={"help": "How many queries to export."})
    case_out: str = field(default="case_study.jsonl")
    case_strategy: str = field(
        default="rank_shift",
        metadata={"help": "rank_shift | gold_missed | low_overlap | random"}
    )
    seed: int = field(default=13)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Data loading
# -------------------------
def _load_retrieval_pickles(data_dir: str, dataset_name: str, split: str) -> Tuple[Dict, Dict, Dict]:
    corpus: Dict = {}
    for corpus_split in ["train", "dev", "test"]:
        if "arguana" in dataset_name:
            read_path = os.path.join(data_dir, f"arguana_{corpus_split}_retrieval_final.pkl")
        else:
            gen_model_name = "gpt4"
            read_path = os.path.join(data_dir, f"{dataset_name}_{corpus_split}_retrieval_{gen_model_name}_final.pkl")

        with open(read_path, "rb") as f:
            split_corpus = pickle.load(f)
        corpus = {**corpus, **split_corpus}

    if "arguana" in dataset_name:
        read_path = os.path.join(data_dir, f"arguana_{split}_retrieval_final.pkl")
    else:
        gen_model_name = "gpt4"
        read_path = os.path.join(data_dir, f"{dataset_name}_{split}_retrieval_{gen_model_name}_final.pkl")

    with open(read_path, "rb") as f:
        _ = pickle.load(f)      # ignore
        queries = pickle.load(f)
        qrels = pickle.load(f)

    return corpus, queries, qrels


def _get_single_gold(qrels_for_q: dict) -> Optional[str]:
    if not qrels_for_q:
        return None
    return max(qrels_for_q.items(), key=lambda x: x[1])[0]


# -------------------------
# Encoding
# -------------------------
@torch.no_grad()
def _encode_texts_hf(model, tokenizer, texts: List[str], max_len: int, batch_size: int, pooler_type: str, device: torch.device) -> torch.Tensor:
    """
    For baseline AutoModel encoders.
    """
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="encode"):
        batch = texts[i:i+batch_size]
        inp = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        out = model(**inp, return_dict=True)

        if pooler_type == "cls":
            x = out.last_hidden_state[:, 0]
        else:
            # avg pool with attention mask
            mask = inp["attention_mask"].unsqueeze(-1).float()
            x = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        x = F.normalize(x, p=2, dim=-1)
        embs.append(x.detach().cpu())

    return torch.cat(embs, dim=0)


@torch.no_grad()
def _encode_texts_ours(
    args: ModelArguments,
    texts: List[str],
    is_query: bool,
    model,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    Match test.py encoding logic for our fine-tuned checkpoints.
    Returns L2-normalized embeddings on CPU float32: (N, dim).
    """
    from torch.utils.data import DataLoader

    dl = DataLoader(texts, batch_size=int(args.batch_size), shuffle=False)

    model.eval()
    outs: List[torch.Tensor] = []

    for batch_texts in tqdm(dl, desc=("EncodeQ" if is_query else "EncodeD")):
        batch_inputs = tokenizer(
            batch_texts,
            max_length=int(args.max_seq_length),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

        base = model.module if hasattr(model, "module") else model

        # our_* checkpoints: use sent_emb=True to get pooler_output
        if isinstance(base, DualBertForCL):
            outputs = model(
                **batch_inputs,
                output_hidden_states=True,
                return_dict=True,
                sent_emb=True,
                is_query=is_query,
            )
            z = outputs.pooler_output
        elif isinstance(base, our_BertForCL) or (NewModelForCL is not None and isinstance(base, NewModelForCL)) or hasattr(base, "sentemb_forward"):
            outputs = model(
                **batch_inputs,
                output_hidden_states=True,
                return_dict=True,
                sent_emb=True,
            )
            z = outputs.pooler_output
        else:
            # If we ever land here for a fine-tuned ckpt, fall back to mean pooling
            outputs = model(**batch_inputs, return_dict=True)
            last_hidden = outputs.last_hidden_state  # (bs, L, H)
            attn = batch_inputs["attention_mask"].unsqueeze(-1)
            z = (last_hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp_min(1e-9)

        # Query-side transform ONLY on queries (exactly as test.py)
        if is_query and args.use_query_transform:
            if hasattr(base, "query_transform") and base.query_transform is not None:
                scale = float(args.query_transform_scale)
                z = z + scale * base.query_transform(z)

        z = F.normalize(z, p=2, dim=-1)
        outs.append(z.cpu())

    return torch.cat(outs, dim=0).to(torch.float32)


def _load_finetuned_model(model_args: ModelArguments, device: torch.device):
    """
    Load your fine-tuned checkpoint using the same logic pattern as test.py.
    """
    model_path = model_args.finetuned_model_name_or_path
    config = AutoConfig.from_pretrained(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # Heuristic: choose correct class
    # - BERT-like: our_BertForCL
    # - DualBertForCL: if your ckpt is dual-encoder
    # - GTE: NewModelForCL (if available)
    arch = (config.architectures[0] if getattr(config, "architectures", None) else "").lower()

    if "dual" in arch:
        model = DualBertForCL.from_pretrained(model_path, from_tf=bool(".ckpt" in model_path), config=config, model_args=model_args)
    elif ("newmodelforcl" in arch) and (NewModelForCL is not None):
        model = NewModelForCL.from_pretrained(model_path, from_tf=bool(".ckpt" in model_path), config=config, model_args=model_args)
    else:
        model = our_BertForCL.from_pretrained(model_path, from_tf=bool(".ckpt" in model_path), config=config, model_args=model_args)

    model = model.to(device)
    if model_args.use_data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model, tokenizer


def _load_baseline_model(name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model = model.to(device)
    model.eval()
    return model, tokenizer


# -------------------------
# FAISS helpers
# -------------------------
def _faiss_index_path(index_dir: str, tag: str, dim: int) -> str:
    os.makedirs(index_dir, exist_ok=True)
    safe_tag = tag.replace("/", "_").replace(":", "_")
    return os.path.join(index_dir, f"flatip_{safe_tag}_d{dim}.faiss")


def _build_or_load_index(vecs: np.ndarray, index_path: str, overwrite: bool) -> faiss.Index:
    dim = vecs.shape[1]
    if (not overwrite) and os.path.exists(index_path):
        logger.info(f"Loading FAISS index: {index_path}")
        return faiss.read_index(index_path)

    logger.info(f"Building FAISS index (IndexFlatIP), dim={dim}, size={vecs.shape[0]}")
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, index_path)
    logger.info(f"Wrote FAISS index: {index_path}")
    return index


# -------------------------
# Case selection + dump
# -------------------------
def _rank_of(docid: Optional[str], doc_ids: List[str], neigh_row: np.ndarray) -> int:
    if docid is None:
        return -1
    for r, di in enumerate(neigh_row):
        if int(di) < 0:
            continue
        if doc_ids[int(di)] == docid:
            return r + 1
    return -1


def _topk_pack(doc_ids: List[str], doc_texts: List[str], neigh_row: np.ndarray, score_row: np.ndarray, topk: int, gold_id: Optional[str]) -> List[dict]:
    out = []
    for r in range(topk):
        di = int(neigh_row[r])
        if di < 0:
            continue
        did = doc_ids[di]
        txt = doc_texts[di].replace("\n", " ").strip()
        out.append({
            "rank": r + 1,
            "docid": did,
            "score": float(score_row[r]),
            "is_gold": bool(gold_id is not None and did == gold_id),
            "snippet": txt[:260],
        })
    return out


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CaseArguments))
    model_args, data_args, case_args = parser.parse_args_into_dataclasses()

    random.seed(case_args.seed)
    np.random.seed(case_args.seed)
    torch.manual_seed(case_args.seed)

    device = _device()
    os.makedirs(case_args.write_path, exist_ok=True)

    # 1) load data
    corpus, queries, qrels = _load_retrieval_pickles(data_args.data_dir, data_args.dataset_name, data_args.split)
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did].get("text", "") for did in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    logger.info(f"Loaded corpus={len(doc_ids)} queries={len(query_ids)}")

    # 2) load models
    ft_model, ft_tok = _load_finetuned_model(model_args, device)
    base_model, base_tok = _load_baseline_model(model_args.baseline_model_name, device)

    # 3) encode docs for BOTH models (two indices)
    logger.info("Encoding docs for fine-tuned model (doc side)...")
    model_args.use_query_transform = False  # doc side should NOT use query transform
    doc_emb_ft = _encode_texts_ours(model_args, doc_texts, is_query=False, model=ft_model, tokenizer=ft_tok, device=device)
    doc_np_ft = doc_emb_ft.numpy().astype("float32")

    logger.info("Encoding docs for baseline model...")
    doc_emb_base = _encode_texts_hf(base_model, base_tok, doc_texts, model_args.max_seq_length, model_args.batch_size, model_args.pooler_type, device)
    doc_np_base = doc_emb_base.numpy().astype("float32")

    # 4) build/load indices
    index_ft_path = _faiss_index_path(case_args.index_dir, f"finetuned_docs_{os.path.basename(model_args.finetuned_model_name_or_path)}", doc_np_ft.shape[1])
    index_base_path = _faiss_index_path(case_args.index_dir, f"baseline_docs_{model_args.baseline_model_name}", doc_np_base.shape[1])
    index_ft = _build_or_load_index(doc_np_ft, index_ft_path, case_args.overwrite_index)
    index_base = _build_or_load_index(doc_np_base, index_base_path, case_args.overwrite_index)

    # 5) encode queries
    logger.info("Encoding queries for fine-tuned model WITH query transform (T(E(q)))...")
    model_args.use_query_transform = True
    q_emb_ft_T = _encode_texts_ours(model_args, query_texts, is_query=True, model=ft_model, tokenizer=ft_tok, device=device)
    q_np_ft_T = q_emb_ft_T.numpy().astype("float32")

    logger.info("Encoding queries for baseline model (E_base(q))...")
    q_emb_base = _encode_texts_hf(base_model, base_tok, query_texts, model_args.max_seq_length, model_args.batch_size, model_args.pooler_type, device)
    q_np_base = q_emb_base.numpy().astype("float32")

    # 6) search
    k = int(case_args.k_neighbors)
    logger.info(f"Searching FAISS with k_neighbors={k} ...")
    scores_T, neigh_T = index_ft.search(q_np_ft_T, k)          # our main target: <T(E_ft(q)), E_ft(d)>
    scores_B, neigh_B = index_base.search(q_np_base, k)        # baseline: <E_base(q), E_base(d)>

    # 7) pick cases
    topk = int(case_args.case_topk)
    cand = []
    for qi, qid in enumerate(query_ids):
        gold = _get_single_gold(qrels.get(qid, {}))
        rT = _rank_of(gold, doc_ids, neigh_T[qi, :k])
        rB = _rank_of(gold, doc_ids, neigh_B[qi, :k])

        if rT == -1:
            continue

        topT = set(doc_ids[int(di)] for di in neigh_T[qi, :topk] if int(di) >= 0 and di != qi)
        topB = set(doc_ids[int(di)] for di in neigh_B[qi, :topk] if int(di) >= 0 and di != qi)
        overlap = len(topT & topB) / float(topk)

        def rc(r):  # rank cost: -1 当作 topk+1
            return (topk + 1) if r == -1 else r

        if case_args.case_strategy == "T_wins":
            rT_c = rc(rT)
            rB_c = rc(rB)

            improve = rB_c - rT_c  # 正数表示 T 更好
            if rT > 10:
                continue
            else:
                score = improve + (1.0 - overlap) * 2.0

        elif case_args.case_strategy == "B_wins":
            rT_c = rc(rT)
            rB_c = rc(rB)
            improve = rT_c - rB_c
            if rB > 10:
                continue
            else:
                score = improve + (1.0 - overlap) * 2.0

        elif case_args.case_strategy == "gold_missed":
            score = (1 if rT == -1 else 0) * 10 + (1 - overlap)

        elif case_args.case_strategy == "low_overlap":
            score = (1 - overlap) * 10 + abs(rc(rT) - rc(rB))

        elif case_args.case_strategy == "random":
            score = random.random()

        else:
            # rank_shift (default)
            score = abs(rc(rT) - rc(rB)) + (1.0 - overlap) * 5.0 + (1 if (rT == -1) != (rB == -1) else 0) * 3.0

        cand.append((score, qi, qid, gold, rT, rB, overlap))

    cand.sort(reverse=True, key=lambda x: x[0])
    picked = cand[:int(case_args.case_max_q)]

    # 8) dump jsonl
    out_path = os.path.join(case_args.write_path, case_args.case_out)
    with open(out_path, "w", encoding="utf-8") as f:
        for _, qi, qid, gold, rT, rB, overlap in picked:
            obj = {
                "qid": qid,
                "query": query_texts[qi],
                "gold_docid": gold,
                "rank_T": rT,
                "rank_baseline": rB,
                "overlap@K": overlap,
                "topK_T": _topk_pack(doc_ids, doc_texts, neigh_T[qi], scores_T[qi], topk, gold),
                "topK_baseline": _topk_pack(doc_ids, doc_texts, neigh_B[qi], scores_B[qi], topk, gold),
                "notes": {
                    "T_scoring": "<T(E_ft(q)), E_ft(d)>",
                    "baseline_scoring": "<E_base(q), E_base(d)>",
                    "baseline_model": model_args.baseline_model_name,
                    "finetuned_ckpt": model_args.finetuned_model_name_or_path,
                }
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    logger.info(f"[case_study] wrote: {out_path}")
    logger.info("Tip: open the jsonl and look for cases with big rank gaps or low overlap.")


if __name__ == "__main__":
    main()
