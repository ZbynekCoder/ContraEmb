import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"

import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, HfArgumentParser

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models

# These are needed to load your trained checkpoint (our_BertForCL / our_gte)
from model.models import our_BertForCL, DualBertForCL
try:
    from model.gte.modeling import NewModelForCL
except Exception:
    NewModelForCL = None


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class ModelArguments:
    # ===== model selection =====
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a fine-tuned checkpoint directory."}
    )
    # legacy alias
    cos_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Deprecated alias of model_name_or_path. If set, it will be used."}
    )
    # zero-shot fallback
    model_name: str = field(
        default="bge",
        metadata={"help": "Base model shortcut if model_name_or_path is not provided: bge|gte|uae."}
    )

    # ===== embedding settings =====
    pooler_type: str = field(default="avg")
    max_seq_length: int = field(default=512)
    batch_size: int = field(default=64)
    use_data_parallel: bool = field(default=False)

    # ===== query-side transform =====
    use_query_transform: bool = field(default=False)
    query_transform_scale: float = field(default=1.0)
    query_transform_dropout: float = field(default=0.1)
    query_transform_init_std: float = field(default=0.02)

    # ===== compatibility fields (IMPORTANT) =====
    # your our_BertForCL __init__ reads these from model_args
    do_mlm: bool = field(default=False)
    mlm_weight: float = field(default=0.1)
    mlp_only_train: bool = field(default=False)

    # some checkpoints/configs expect these to exist
    temp: float = field(default=0.02)
    hard_negative_weight: float = field(default=0.0)
    loss_type: str = field(default="cos")


@dataclass
class EvalArguments:
    dataset_name: str = field(default="arguana")
    split: str = field(default="dev", metadata={"help": "dev | test"})
    write_path: str = field(default="test_results/cos_only")
    k_neighbors: int = field(default=1000)
    index_dir: str = field(default="./indices")
    overwrite_index: bool = field(default=False)

    # ===== time benchmark (Xu Appendix-F style) =====
    do_time_benchmark: bool = field(
        default=False,
        metadata={"help": "If True: only run vector-time benchmark (100 queries x 100 docs) and exit."}
    )
    num_time_queries: int = field(default=100, metadata={"help": "Number of queries for timing."})
    num_time_docs: int = field(default=100, metadata={"help": "Number of docs to score per query."})
    warmup: int = field(default=10, metadata={"help": "Warmup iterations (not timed)."})
    seed: int = field(default=123, metadata={"help": "RNG seed for selecting doc subset."})
    emb_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "If set: save/load preprocessed embeddings (doc.npy/query.npy/doc_ids.pkl/query_ids.pkl)."}
    )


def _ensure_model_args_compat(model_args: ModelArguments) -> None:
    """Make eval robust even if dataclass changes."""
    defaults = {
        "do_mlm": False,
        "mlm_weight": 0.1,
        "mlp_only_train": False,
        "temp": 0.02,
        "hard_negative_weight": 0.0,
        "loss_type": "cos",
    }
    for k, v in defaults.items():
        if not hasattr(model_args, k):
            setattr(model_args, k, v)


def _select_model_path(args: ModelArguments) -> Optional[str]:
    if args.cos_model_name_or_path:
        if args.model_name_or_path and args.model_name_or_path != args.cos_model_name_or_path:
            logger.warning(
                "Both model_name_or_path and cos_model_name_or_path are set; "
                "using cos_model_name_or_path."
            )
        return args.cos_model_name_or_path
    return args.model_name_or_path


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_retrieval_pickles(dataset_name: str, split: str) -> Tuple[Dict, Dict, Dict]:
    """
    Expected pickle format (same convention as your original script):
      - corpus pkls: pickle.load(f) -> corpus dict
      - split pkl: three objects: (ignored), queries, qrels
    """
    corpus: Dict = {}
    for corpus_split in ["train", "dev", "test"]:
        if "arguana" in dataset_name:
            read_path = f"./data/arguana_{corpus_split}_retrieval_final.pkl"
        else:
            gen_model_name = "gpt4"
            read_path = f"./data/{dataset_name}_{corpus_split}_retrieval_{gen_model_name}_final.pkl"

        with open(read_path, "rb") as f:
            split_corpus = pickle.load(f)
        corpus = {**corpus, **split_corpus}

    if "arguana" in dataset_name:
        read_path = f"./data/arguana_{split}_retrieval_final.pkl"
    else:
        gen_model_name = "gpt4"
        read_path = f"./data/{dataset_name}_{split}_retrieval_{gen_model_name}_final.pkl"

    with open(read_path, "rb") as f:
        _ = pickle.load(f)      # ignore
        queries = pickle.load(f)
        qrels = pickle.load(f)

    return corpus, queries, qrels


def _load_encoder(args: ModelArguments):
    """
    Loads either:
      - fine-tuned checkpoint (our_BertForCL / NewModelForCL) if model_name_or_path is set
      - else a base AutoModel (bge/gte/uae)
    """
    _ensure_model_args_compat(args)

    device = _device()
    model_path = _select_model_path(args)

    if model_path:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if getattr(config, "dual_encoder", False):
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = DualBertForCL.from_pretrained(
                model_path,
                config=config,
                model_args=args,
            )
            model = model.to(device)
            if args.use_data_parallel and torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            return model, tokenizer

        lower = model_path.lower()
        if (("gte" in lower) or ("our_gte" in lower)) and (NewModelForCL is not None):
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = NewModelForCL.from_pretrained(
                model_path,
                model_args=args,
                config=config,
                add_pooling_layer=True,
                trust_remote_code=True,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = our_BertForCL.from_pretrained(
                model_path,
                from_tf=bool(".ckpt" in model_path),
                config=config,
                model_args=args,
            )

        model = model.to(device)
        if args.use_data_parallel and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        return model, tokenizer

    # Zero-shot fallback
    if args.model_name.lower() == "bge":
        name = "BAAI/bge-base-en-v1.5"
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
    elif args.model_name.lower() == "gte":
        name = "Alibaba-NLP/gte-large-en-v1.5"
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModel.from_pretrained(name, trust_remote_code=True)
    elif args.model_name.lower() == "uae":
        name = "WhereIsAI/UAE-Large-V1"
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModel.from_pretrained(name, trust_remote_code=True)
    else:
        raise ValueError(f"Unknown model_name={args.model_name}. Use bge|gte|uae or set --model_name_or_path.")

    model = model.to(device)
    if args.use_data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model, tokenizer


@torch.no_grad()
def _encode_texts(
    args: ModelArguments,
    texts: List[str],
    is_query: bool,
    model,
    tokenizer,
) -> torch.Tensor:
    """
    Returns L2-normalized embeddings on CPU float32: (N, dim).
    """
    device = _device()
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
                is_query=is_query
            )
            z = outputs.pooler_output
        elif isinstance(base, our_BertForCL) or (NewModelForCL is not None and isinstance(base, NewModelForCL)) or hasattr(base, "sentemb_forward"):
            outputs = model(
                **batch_inputs,
                output_hidden_states=True,
                return_dict=True,
                sent_emb=True
            )
            z = outputs.pooler_output
        else:
            # base models: mean pooling
            outputs = model(**batch_inputs, return_dict=True)
            last_hidden = outputs.last_hidden_state  # (bs, L, H)
            attn = batch_inputs["attention_mask"].unsqueeze(-1)
            z = (last_hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp_min(1e-9)

        # Query-side transform ONLY on queries
        if is_query and args.use_query_transform:
            if hasattr(base, "query_transform") and base.query_transform is not None:
                scale = float(args.query_transform_scale)
                z = z + scale * base.query_transform(z)

        z = F.normalize(z, p=2, dim=-1)
        outs.append(z.cpu())

    return torch.cat(outs, dim=0).to(torch.float32)


def _build_or_load_index(
    doc_np: np.ndarray,
    dim: int,
    index_file: str,
    overwrite: bool = False,
) -> faiss.Index:
    """
    Use HNSW inner product index (cosine for normalized vectors).
    """
    if (not overwrite) and os.path.exists(index_file):
        index = faiss.read_index(index_file)
        logger.info("Loaded FAISS index: %s", index_file)
        return index

    index = faiss.IndexHNSWFlat(dim, 64, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(doc_np)
    faiss.write_index(index, index_file)
    logger.info("Built & saved FAISS index: %s", index_file)
    return index


def _cache_paths(cache_dir: str) -> Dict[str, str]:
    return {
        "doc": os.path.join(cache_dir, "doc.npy"),
        "query": os.path.join(cache_dir, "query.npy"),
        "doc_ids": os.path.join(cache_dir, "doc_ids.pkl"),
        "query_ids": os.path.join(cache_dir, "query_ids.pkl"),
        "meta": os.path.join(cache_dir, "meta.txt"),
    }


def _maybe_load_cached_embeddings(eval_args: EvalArguments):
    if not eval_args.emb_cache_dir:
        return None
    cache_dir = eval_args.emb_cache_dir
    paths = _cache_paths(cache_dir)
    if all(os.path.exists(paths[k]) for k in ["doc", "query", "doc_ids", "query_ids"]):
        doc_np = np.load(paths["doc"]).astype("float32")
        query_np = np.load(paths["query"]).astype("float32")
        with open(paths["doc_ids"], "rb") as f:
            doc_ids = pickle.load(f)
        with open(paths["query_ids"], "rb") as f:
            query_ids = pickle.load(f)
        logger.info("Loaded cached embeddings from: %s", cache_dir)
        return doc_np, query_np, doc_ids, query_ids
    return None


def _save_cached_embeddings(eval_args: EvalArguments, doc_np, query_np, doc_ids, query_ids, model_args: ModelArguments):
    if not eval_args.emb_cache_dir:
        return
    cache_dir = eval_args.emb_cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    paths = _cache_paths(cache_dir)
    np.save(paths["doc"], doc_np)
    np.save(paths["query"], query_np)
    with open(paths["doc_ids"], "wb") as f:
        pickle.dump(doc_ids, f)
    with open(paths["query_ids"], "wb") as f:
        pickle.dump(query_ids, f)
    with open(paths["meta"], "w", encoding="utf-8") as f:
        f.write(f"created_at={datetime.now().isoformat()}\n")
        f.write(f"dataset={eval_args.dataset_name}\n")
        f.write(f"split={eval_args.split}\n")
        f.write(f"max_seq_length={model_args.max_seq_length}\n")
        f.write(f"batch_size={model_args.batch_size}\n")
        f.write(f"use_query_transform={model_args.use_query_transform}\n")
        f.write(f"query_transform_scale={model_args.query_transform_scale}\n")
        f.write(f"model_path={_select_model_path(model_args)}\n")
        f.write(f"model_name={model_args.model_name}\n")
    logger.info("Saved cached embeddings to: %s", cache_dir)


def _run_time_benchmark(
    eval_args: EvalArguments,
    doc_np: np.ndarray,
    query_np: np.ndarray,
) -> Dict[str, float]:
    """
    Xu Appendix-F style:
    - Assume embeddings are preprocessed
    - Compute vector score between 1 query embedding and 100 doc embeddings
    - Report average running time for 100 queries
    Here 'score' is cosine/dot-product because vectors are L2-normalized.
    """
    rng = np.random.default_rng(int(eval_args.seed))
    n_docs = int(eval_args.num_time_docs)
    n_q = int(eval_args.num_time_queries)

    if len(doc_np) < n_docs:
        raise ValueError(f"Corpus too small: have {len(doc_np)} docs but need num_time_docs={n_docs}")
    if len(query_np) < n_q:
        raise ValueError(f"Queries too small: have {len(query_np)} queries but need num_time_queries={n_q}")

    doc_idx = rng.choice(len(doc_np), size=n_docs, replace=False)
    D = doc_np[doc_idx]  # (n_docs, dim)
    Q = query_np[:n_q]   # (n_q, dim)

    # Warmup (not timed)
    w = min(int(eval_args.warmup), n_q)
    for i in range(w):
        _ = Q[i] @ D.T

    # Timed section: 100 queries, each scoring against 100 docs
    t0 = time.perf_counter()
    for i in range(n_q):
        _ = Q[i] @ D.T
    t1 = time.perf_counter()

    total_s = float(t1 - t0)
    per_query_s = total_s / float(n_q)
    per_query_ms = per_query_s * 1000.0
    qps = 1.0 / per_query_s if per_query_s > 0 else float("inf")

    return {
        "total_s": total_s,
        "per_query_ms": per_query_ms,
        "qps": qps,
        "num_time_queries": float(n_q),
        "num_time_docs": float(n_docs),
    }


def main():
    parser = HfArgumentParser((ModelArguments, EvalArguments))
    model_args, eval_args = parser.parse_args_into_dataclasses()

    os.makedirs(eval_args.write_path, exist_ok=True)

    # If timing-only, use a different filename to avoid confusion
    suffix = "time_benchmark" if eval_args.do_time_benchmark else "cos_only"
    out_path = os.path.join(
        eval_args.write_path,
        f"{eval_args.dataset_name}_{eval_args.split}_{suffix}.txt"
    )

    with open(out_path, "w", encoding="utf-8") as out_f:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] dataset={eval_args.dataset_name} split={eval_args.split}", file=out_f)
        logger.info("dataset=%s split=%s", eval_args.dataset_name, eval_args.split)

        # Load data (needed for ids/texts unless cache already has everything)
        corpus, queries, qrels = _load_retrieval_pickles(eval_args.dataset_name, eval_args.split)

        # flatten corpus/queries
        doc_ids: List[str] = []
        doc_texts: List[str] = []
        for pid, value in corpus.items():
            doc_ids.append(pid)
            doc_texts.append(value.get("text", "").strip())

        query_ids: List[str] = []
        query_texts: List[str] = []
        for qid, text in queries.items():
            query_ids.append(qid)
            query_texts.append(text.strip())

        print(f"Corpus size = {len(doc_ids)} docs, Query size = {len(query_ids)}", file=out_f)
        logger.info("Corpus size=%d, Query size=%d", len(doc_ids), len(query_ids))

        # Try load cached embeddings first (to match "preprocessed embeddings" setting)
        cached = _maybe_load_cached_embeddings(eval_args)
        if cached is not None:
            doc_np, query_np, doc_ids_cached, query_ids_cached = cached
            # Prefer cached ids to keep consistent indexing
            doc_ids = doc_ids_cached
            query_ids = query_ids_cached
            dim = int(doc_np.shape[1])
        else:
            # load encoder
            model, tokenizer = _load_encoder(model_args)

            # encode
            doc_emb = _encode_texts(model_args, doc_texts, is_query=False, model=model, tokenizer=tokenizer)
            query_emb = _encode_texts(model_args, query_texts, is_query=True, model=model, tokenizer=tokenizer)

            doc_np = doc_emb.numpy().astype("float32")
            query_np = query_emb.numpy().astype("float32")
            dim = int(doc_np.shape[1])

            _save_cached_embeddings(eval_args, doc_np, query_np, doc_ids, query_ids, model_args)

        # ===== Timing-only path =====
        if eval_args.do_time_benchmark:
            bench = _run_time_benchmark(eval_args, doc_np, query_np)
            print("=== Vector Time Benchmark (Xu Appendix-F style) ===", file=out_f)
            print(f"Assume embeddings preprocessed: {bool(eval_args.emb_cache_dir)}", file=out_f)
            print(f"num_queries={int(bench['num_time_queries'])}, num_docs_per_query={int(bench['num_time_docs'])}", file=out_f)
            print(f"total_time_s={bench['total_s']:.6f}", file=out_f)
            print(f"avg_time_per_query_ms={bench['per_query_ms']:.6f}", file=out_f)
            print(f"qps={bench['qps']:.2f}", file=out_f)
            logger.info("Timing done. Saved to: %s", out_path)
            return

        # ===== Original retrieval+eval path (unchanged) =====
        os.makedirs(eval_args.index_dir, exist_ok=True)
        model_path = _select_model_path(model_args)
        if model_path:
            idx_tag = os.path.basename(os.path.dirname(os.path.normpath(model_path)))
        else:
            idx_tag = model_args.model_name.lower()
        index_file = os.path.join(eval_args.index_dir, f"{eval_args.dataset_name}-{idx_tag}-ip-hnsw.faiss")

        index = _build_or_load_index(doc_np, dim, index_file, overwrite=bool(eval_args.overwrite_index))
        index.hnsw.efSearch = max(int(eval_args.k_neighbors), 128)

        k = int(eval_args.k_neighbors)
        scores, neighbors = index.search(query_np, k)

        results: Dict[str, Dict[str, float]] = {}
        for qi, qid in enumerate(query_ids):
            res_q: Dict[str, float] = {}
            for rank in range(k):
                di = int(neighbors[qi, rank])
                if di < 0:
                    continue
                res_q[doc_ids[di]] = float(scores[qi, rank])
            results[qid] = res_q

        dummy = DRES(models.SentenceBERT("BAAI/bge-base-en-v1.5"), batch_size=16)
        retriever = EvaluateRetrieval(dummy, score_function="cos_sim")
        k_values = [1, 3, 5, 10, 100, 1000]
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)

        print("NDCG:", ndcg, file=out_f)
        print("MAP:", _map, file=out_f)
        print("Recall:", recall, file=out_f)
        print("Precision:", precision, file=out_f)

        logger.info("NDCG=%s", ndcg)
        logger.info("MAP=%s", _map)
        logger.info("Recall=%s", recall)
        logger.info("Precision=%s", precision)
        logger.info("Saved eval to: %s", out_path)


if __name__ == "__main__":
    main()
