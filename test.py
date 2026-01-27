import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"

import logging
import pickle
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
from sparsecl.models import our_BertForCL, DualBertForCL
try:
    from sparsecl.gte.modeling import NewModelForCL
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

        # Heuristic: if it's a GTE-style checkpoint and NewModelForCL exists
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
            outputs = model(**batch_inputs,
                            output_hidden_states=True,
                            return_dict=True,
                            sent_emb=True,
                            is_query=is_query)
            z = outputs.pooler_output
        elif isinstance(base, our_BertForCL) or (NewModelForCL is not None and isinstance(base, NewModelForCL)) or hasattr(base, "sentemb_forward"):
            outputs = model(**batch_inputs,
                            output_hidden_states=True,
                            return_dict=True,
                            sent_emb=True)
            z = outputs.pooler_output
        elif isinstance(base, our_BertForCL) or (NewModelForCL is not None and isinstance(base, NewModelForCL)) or hasattr(base, "sentemb_forward"):
            outputs = model(**batch_inputs, output_hidden_states=True, return_dict=True, sent_emb=True)
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


def main():
    parser = HfArgumentParser((ModelArguments, EvalArguments))
    model_args, eval_args = parser.parse_args_into_dataclasses()

    # output files
    os.makedirs(eval_args.write_path, exist_ok=True)
    out_path = os.path.join(eval_args.write_path, f"{eval_args.dataset_name}_{eval_args.split}_cos_only.txt")
    with open(out_path, "w", encoding="utf-8") as out_f:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] dataset={eval_args.dataset_name} split={eval_args.split}", file=out_f)
        logger.info("dataset=%s split=%s", eval_args.dataset_name, eval_args.split)

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

        print(f"Corpus size = {len(doc_ids)} docs, Query size = {len(query_ids)}")

        # load encoder
        model, tokenizer = _load_encoder(model_args)

        # encode
        doc_emb = _encode_texts(model_args, doc_texts, is_query=False, model=model, tokenizer=tokenizer)
        query_emb = _encode_texts(model_args, query_texts, is_query=True, model=model, tokenizer=tokenizer)

        doc_np = doc_emb.numpy().astype("float32")
        query_np = query_emb.numpy().astype("float32")
        dim = int(doc_np.shape[1])

        # index filename tags
        os.makedirs(eval_args.index_dir, exist_ok=True)
        model_path = _select_model_path(model_args)
        idx_tag = os.path.basename(os.path.dirname(os.path.normpath(model_path)))
        index_file = os.path.join(eval_args.index_dir, f"{eval_args.dataset_name}-{idx_tag}-ip-hnsw.faiss")

        index = _build_or_load_index(doc_np, dim, index_file, overwrite=bool(eval_args.overwrite_index))
        index.hnsw.efSearch = max(int(eval_args.k_neighbors), 128)

        # search
        k = int(eval_args.k_neighbors)
        scores, neighbors = index.search(query_np, k)

        # build BEIR results dict
        results: Dict[str, Dict[str, float]] = {}
        for qi, qid in enumerate(query_ids):
            res_q: Dict[str, float] = {}
            for rank in range(k):
                di = int(neighbors[qi, rank])
                if di < 0:
                    continue
                res_q[doc_ids[di]] = float(scores[qi, rank])
            results[qid] = res_q

        # Evaluate with BEIR helper (model object is unused for metrics, but API needs one)
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
