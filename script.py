import torch
from sparsecl.models import our_BertForCL
from transformers import AutoConfig

MODEL_PATH = "results/our-bge-arguana-qr-phase1"

class ModelArgs:
    content_head_dim = 768
    stance_head_dim = 128
    temp = 0.05
    pooler_type = "avg"
    do_mlm = False

config = AutoConfig.from_pretrained(MODEL_PATH)
model = our_BertForCL.from_pretrained(MODEL_PATH, config=config, model_args=ModelArgs())

# 检查 content_head 的第一个参数
weight_sum = model.content_head.dense.weight.sum().item()
print(f"Content Head Weight Sum: {weight_sum}")

if weight_sum == 0 or abs(weight_sum) < 1e-5: # 或者是某个特定常数
    print("❌ 警告：权重看起来像是随机初始化的（或者是空的）。")
else:
    print("✅ 权重看起来已经成功加载（非零且有数值）。")
