import torch
import pandas as pd
import os
import re
import csv
from PIL import Image
from transformers import AutoModel
import json

# ✅ 设备选择
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ✅ 定义 BGE 模型名称
MODEL_NAME = "BAAI/BGE-VL-MLLM-S2"

# ✅ 加载 BGE 模型
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
model.eval()

# ✅ 加载数据
passage_file = "challenge_passage/train-00000-of-00001.parquet"
query_file = "challenge_data/train-00000-of-00001.parquet"
df_passages = pd.read_parquet(passage_file)
df_queries = pd.read_parquet(query_file)

# ✅ 图片目录
image_dir = "images/"

# ✅ 过滤无效字符的函数
def clean_text(text):
    if not isinstance(text, str):
        return "[EMPTY]"
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "[EMPTY]"

# ✅ 检查图片有效性
def check_image_valid(image_path):
    full_path = os.path.join(image_dir, image_path)
    if os.path.exists(full_path):
        try:
            img = Image.open(full_path)
            img.verify()
            img = Image.open(full_path).convert("RGB")
            return full_path
        except Exception as e:
            print(f"[ERROR] 无法读取图片: {full_path}, 错误: {e}")
    return None

# ✅ 仅处理指定范围的 query
batch_queries = df_queries.iloc[1:2420]
output_file = "submission_bge_m2kr_part1_2420.csv"

# 创建 CSV 并写入标题
pd.DataFrame(columns=["question_id", "passage_id"]).to_csv(output_file, index=False, encoding="utf-8")

with torch.no_grad():
    model.set_processor(MODEL_NAME)
    model.processor.patch_size = 14

    for idx, query in batch_queries.iterrows():
        question_id = query["question_id"]
        instruction = query["instruction"]
        question = query["question"] if pd.notna(query["question"]) else None
        img_path = query["img_path"] if pd.notna(query["img_path"]) else None
        print(f"[INFO] 处理 question_id: {question_id} ({idx + 1}/2420)")

        # **计算查询文本的 embedding**
        query_text = instruction if question is None else f"{instruction} {question}"
        query_inputs = model.data_process(text=query_text, images=None, q_or_c="q")
        query_embs = model(**query_inputs, output_hidden_states=True)[:, -1, :].to(device)
        query_embs = torch.nn.functional.normalize(query_embs, dim=-1)

        # **计算候选 passage 的 embedding**
        candidate_texts = df_passages["passage_content"].apply(clean_text).tolist()
        candidate_images = df_passages["page_screenshot"].apply(check_image_valid).tolist()
        passage_ids = df_passages["passage_id"].tolist()

        batch_size = 1
        candi_embs_list = []
        num_batches = len(candidate_texts) // batch_size + (1 if len(candidate_texts) % batch_size > 0 else 0)

        for i in range(num_batches):
            batch_texts = candidate_texts[i * batch_size: (i + 1) * batch_size]
            batch_images = candidate_images[i * batch_size: (i + 1) * batch_size]

            try:
                torch.cuda.empty_cache()
                candidate_inputs = model.data_process(text=batch_texts, images=batch_images, q_or_c="c")
                batch_candi_embs = model(**candidate_inputs, output_hidden_states=True)[:, -1, :].to(device)
                batch_candi_embs = torch.nn.functional.normalize(batch_candi_embs, dim=-1)
                candi_embs_list.append(batch_candi_embs)
            except Exception as e:
                print(f"[ERROR] 处理图片失败: {e}")
                torch.cuda.empty_cache()
                continue

        if candi_embs_list:
            candi_embs = torch.cat(candi_embs_list, dim=0)
        else:
            print(f"[ERROR] 无可用候选项 for question_id: {question_id}")
            continue

        # **计算相似度**
        scores = torch.matmul(query_embs, candi_embs.T)

        # **获取前5个最佳匹配 passage_id**
        top_k = 5
        top_indices = torch.topk(scores, top_k, dim=1).indices.squeeze(0).tolist()
        top_passage_ids = [passage_ids[idx] for idx in top_indices]

        # **保存结果到 CSV 文件**
        pd.DataFrame([[question_id, str(top_passage_ids)]], columns=["question_id", "passage_id"]).to_csv(
            output_file, mode='a', header=False, index=False, encoding="utf-8"
        )

print(f"✅ 处理完成，查询 1 到 2420 结果已保存到 {output_file}")
