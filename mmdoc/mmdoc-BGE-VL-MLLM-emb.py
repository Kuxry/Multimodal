import os
import re
import csv
import json
import io
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoModel

# ---------- 设备选择 ----------
# 指定使用 GPU 1 和 GPU 2（请确保这两张 GPU 空闲）
device_ids = [1, 2]
primary_device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

# ---------- 1. 加载 BAAI/BGE-VL-MLLM-S2 模型 ----------
MODEL_NAME = "BAAI/BGE-VL-MLLM-S2"
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
model.to(primary_device)  # 模型加载到主设备

with torch.no_grad():
    model.set_processor(MODEL_NAME)

# 使用 DataParallel 包装模型以利用多卡
model = torch.nn.DataParallel(model, device_ids=device_ids)
print(f"✅ BAAI/BGE-VL-MLLM-S2 模型加载完成，并使用 GPU {device_ids} 进行加速！")

# ---------- 2. 读取文档库（MMDocIR_doc_passages.parquet） ----------
df_passages = pd.read_parquet("MMDocIR_doc_passages.parquet")
print(f"✅ 文档库加载完成，共 {len(df_passages)} 条记录！")

# ---------- 3. 计算文档嵌入 ----------
# 对于每条记录，优先使用 vlm_text，其次 ocr_text；若两者均不存在，则使用图像数据
passage_ids = []
embeddings_list = []

for idx, passage in df_passages.iterrows():
    passage_id = passage["passage_id"]

    # 获取文本信息
    vlm_text = passage.get("vlm_text", "").strip() if "vlm_text" in passage else ""
    ocr_text = passage.get("ocr_text", "").strip() if "ocr_text" in passage else ""

    if vlm_text:
        # 调用 data_process 时需通过 model.module 访问原始模型
        input_data = model.module.data_process(
            text=vlm_text,
            q_or_c="c"
        )
    elif ocr_text:
        input_data = model.module.data_process(
            text=ocr_text,
            q_or_c="c"
        )
    else:
        # 当文本信息为空时，使用图像数据
        try:
            image_data = passage["image_binary"]
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            print(f"❌ 处理 passage_id {passage_id} 的图像时出错: {e}")
            continue
        input_data = model.module.data_process(
            images=image,
            q_or_c="c"
        )

    with torch.no_grad():
        # 获取最后一层隐藏状态的最后一个 token 作为嵌入
        embedding = model(**input_data, output_hidden_states=True)[:, -1, :]
    embedding = torch.nn.functional.normalize(embedding, dim=-1)

    passage_ids.append(passage_id)
    embeddings_list.append(embedding.cpu())

# 将所有嵌入拼接为一个张量（形状：[num_passages, emb_dim]）
if embeddings_list:
    embeddings_tensor = torch.cat(embeddings_list, dim=0)
else:
    embeddings_tensor = torch.empty(0)

# ---------- 4. 保存为 npy 格式 ----------
# 将 tensor 转换为 NumPy 数组后分别保存嵌入和 passage_id
embeddings_np = embeddings_tensor.numpy()
ids_np = np.array(passage_ids)

np.save("mmdoc_emb.npy", embeddings_np)
np.save("mmdoc_ids.npy", ids_np)
print("✅ 嵌入计算完成，已保存到 mmdoc_emb.npy 和 mmdoc_ids.npy")
