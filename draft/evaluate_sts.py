import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
import os
import sys

# 确保能引用到项目中的 sparsecl 模块
sys.path.append(os.getcwd())
from sparsecl.models import our_BertForCL

# ==========================================
# ⚙️ 配置区域 (请根据你的实际路径调整)
# ==========================================
# 1. 填入你 Phase 1 训练完的模型路径
MODEL_PATH = "../results/our-bge-arguana-qr-phase1"


# 2. 填入你训练时使用的参数 (必须完全一致，否则 Head 权重加载会错位)
class ModelArgs:
    content_head_dim = 768
    stance_head_dim = 128
    temp = 0.05
    pooler_type = "avg"  # 你之前改回了 avg，请确认
    do_mlm = False
    mlm_weight = 0.1


# ==========================================
# 🚀 1. 加载模型与数据
# ==========================================
print(f"[*] 正在从 {MODEL_PATH} 加载模型...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
config = AutoConfig.from_pretrained(MODEL_PATH)

# 使用自定义的 model_args 初始化
model = our_BertForCL.from_pretrained(
    MODEL_PATH,
    config=config,
    model_args=ModelArgs()
)
model.to(device)
model.eval()

print("[*] 正在加载 STS-B 验证集...")
# 加载 HuggingFace 上的 STS-B 数据集 (验证集)
dataset = load_dataset("sentence-transformers/stsb", split="validation")


# ==========================================
# 🧠 2. 提取 Content 嵌入函数
# ==========================================
def get_content_embeddings(texts, batch_size=64):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            # 关键：调用 sent_emb=True 获取解耦后的字典
            embeddings_dict = model(**inputs, sent_emb=True)
            # 我们只取 content (a 向量)
            content_vecs = embeddings_dict["content"].cpu().numpy()
            all_embeddings.append(content_vecs)

    return np.vstack(all_embeddings)


# ==========================================
# 📏 3. 执行评估
# ==========================================
sentences1 = dataset["sentence1"]
sentences2 = dataset["sentence2"]
gold_scores = dataset["score"]

print(f"[*] 正在为 {len(sentences1)} 对句子生成 Content 嵌入...")

emb1 = get_content_embeddings(sentences1)
emb2 = get_content_embeddings(sentences2)


# 计算余弦相似度
def cosine_sim(a, b):
    # 先归一化
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a_norm * b_norm, axis=1)


predicted_sims = cosine_sim(emb1, emb2)

# 计算相关系数
pearson_corr, _ = pearsonr(predicted_sims, gold_scores)
spearman_corr, _ = spearmanr(predicted_sims, gold_scores)

# ==========================================
# 🏁 4. 结果输出
# ==========================================
print("\n" + "=" * 40)
print("📊 STS-B 评估结果 (针对 Content Head)")
print("=" * 40)
print(f"Pearson Correlation  : {pearson_corr:.4f}")
print(f"Spearman Correlation : {spearman_corr:.4f}")
print("=" * 40)

if spearman_corr > 0.8:
    print("✅ 结果非常棒！Content 空间语义保存完美。")
elif spearman_corr > 0.7:
    print("🆗 结果还可以。虽然略有下降，但基础语义还在。")
else:
    print("⚠️ 警告：Content 语义受损严重，建议检查 Phase 1 学习率。")
