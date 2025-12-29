import torch
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import ndcg_score
from sparsecl.models import our_BertForCL
from transformers import AutoTokenizer, AutoConfig


def calculate_ndcg(q_vecs, d_vecs, qrels, k=10):
    """ 计算 NDCG@10 """
    scores = []
    for q_id, query_vec in enumerate(q_vecs):
        # 计算该 Query 对所有候选文档的相似度
        sim_scores = torch.matmul(d_vecs, query_vec.unsqueeze(1)).squeeze().cpu().numpy()

        # 构建真值标签 (Ground Truth)
        # qrels[q_id] 应该是一个数组，矛盾文档位置为 1，其余为 0
        labels = qrels[q_id]
        scores.append(ndcg_score([labels], [sim_scores], k=k))
    return np.mean(scores)


@torch.no_grad()
def evaluate_witt_model(model_path, data_path, device="cuda", alpha=1.0):
    # 1. 加载模型与配置
    config = AutoConfig.from_pretrained(model_path)
    if not hasattr(config, 'do_mlm'):
        config.do_mlm = False
    if not hasattr(config, 'temp'):
        config.temp = 0.05
    if not hasattr(config, 'mlp_only_train'):
        config.mlp_only_train = False
    config.pooler_type = "avg"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = our_BertForCL.from_pretrained(model_path, config=config, model_args=config)
    model.to(device).eval()

    # 2. 加载数据 (假设你已经有处理好的 Arguana pickle)
    with open(data_path, "rb") as f:
        # 这里的加载逻辑需对应你数据脚本的 dump 顺序
        corpus_data = pickle.load(f)  # {id: text}
        queries_data = pickle.load(f)  # {id: text}
        qrels_data = pickle.load(f)  # {q_id: {d_id: 1}}

    # 3. 提取特征嵌入 (Embedding Extraction)
    print(">>> Extracting Embeddings...")
    q_ids = list(queries_data.keys())
    d_ids = list(corpus_data.keys())

    # 存储 a 向量和 b 向量
    q_a, q_b = [], []
    d_a, d_b = [], []

    for i, q_id in enumerate(tqdm(q_ids, desc="Encoding Queries")):
        inputs = tokenizer(queries_data[q_id], return_tensors="pt", truncation=True, max_length=512).to(device)
        out = model(**inputs, sent_emb=True)  # 调用你的 sentemb_forward

        # [🔍 CHECK] 打印前 5 个样本的模长
        if i < 5:
            # 计算 L2 Norm
            norm_a = torch.norm(out["content"], p=2, dim=-1).item()
            norm_b = torch.norm(out["stance"], p=2, dim=-1).item()
            
            # 理论最大模长 (对于 Tanh 激活)
            # content_dim=768 -> sqrt(768) ≈ 27.7
            # stance_dim=128  -> sqrt(128) ≈ 11.3
            print(f"\n[Debug Sample {i}]")
            print(f"  Norm(a) [Content]: {norm_a:.4f}")
            print(f"  Norm(b) [Stance] : {norm_b:.4f}  <-- 关注这个！如果接近 11.3 说明饱和了")
            
        q_a.append(torch.nn.functional.normalize(out["content"], p=2, dim=-1))
        q_b.append(torch.nn.functional.normalize(out["stance"], p=2, dim=-1))

    for d_id in tqdm(d_ids, desc="Encoding Corpus"):
        raw_input = corpus_data[d_id]

        # 自动处理字典类型 (Arguana Corpus 通常包含 title 和 text)
        if isinstance(raw_input, dict):
            # 将标题和正文拼接，这是检索任务的标准做法
            title = raw_input.get("title", "")
            body = raw_input.get("text", "")
            text = (title + " " + body).strip()
        else:
            text = str(raw_input)

        # 增加一个空值保护
        if not text:
            text = "empty document"

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        out = model(**inputs, sent_emb=True)
        d_a.append(torch.nn.functional.normalize(out["content"], p=2, dim=-1))
        d_b.append(torch.nn.functional.normalize(out["stance"], p=2, dim=-1))


    Q_A, Q_B = torch.cat(q_a), torch.cat(q_b)
    D_A, D_B = torch.cat(d_a), torch.cat(d_b)

    # 4. 构造 Label 矩阵
    labels_matrix = np.zeros((len(q_ids), len(d_ids)))
    for i, q_id in enumerate(q_ids):
        for d_id, rel in qrels_data[q_id].items():
            if d_id in d_ids:
                labels_matrix[i, d_ids.index(d_id)] = rel

    # 5. 核心评测逻辑
    print("\n" + "=" * 30)
    print("🏆 WITT DECOUPLING EVALUATION")
    print("=" * 30)

    # A. Content-Only (仅靠内容)
    ndcg_a = calculate_ndcg(Q_A, D_A, labels_matrix)
    print(f"🔹 Content-Only NDCG@10 (a): {ndcg_a:.4f}")

    # B. Stance-Only (仅靠立场)
    ndcg_b = calculate_ndcg(Q_B, D_B, labels_matrix)
    print(f"🔹 Stance-Only  NDCG@10 (b): {ndcg_b:.4f}")

    # C. Witt-Decoupled (矛盾检索公式: Content - alpha * Stance)
    # 逻辑：我们要找话题相同 (A高) 但立场相反 (B低) 的文档
    combined_scores = []
    for i in range(len(q_ids)):
        score_a = torch.matmul(D_A, Q_A[i].unsqueeze(1)).squeeze()
        score_b = torch.matmul(D_B, Q_B[i].unsqueeze(1)).squeeze()
        # 核心：寻找 B 相似度最低的作为矛盾
        final_score = score_a - alpha * score_b
        combined_scores.append(ndcg_score([labels_matrix[i]], [final_score.cpu().numpy()], k=10))

    print(f"🔥 Witt-Combined NDCG@10 (a - {alpha}b): {np.mean(combined_scores):.4f}")
    print("=" * 30)


if __name__ == "__main__":
    # 修改为你的模型路径和数据路径
    evaluate_witt_model(
        model_path="../results/repository/phase2/202512230039/our-bge-arguana-qr-phase2",
        data_path="../data/arguana_test_retrieval_final.pkl",
        alpha=0.05  # 这个参数可以动态调优
    )
