import torch
import pandas as pd
import os
import re
from PIL import Image
from transformers import AutoModel
import csv
import warnings
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# ----------------------------------------------------
# 1. 设备选择
# ----------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ----------------------------------------------------
# 2. 定义 BGE 模型名称 & 加载模型
# ----------------------------------------------------
MODEL_NAME = "BAAI/BGE-VL-MLLM-S2"
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
model.eval()

# ----------------------------------------------------
# 3. 加载数据（Passage & Query）
# ----------------------------------------------------
passage_file = "challenge_passage/train-00000-of-00001.parquet"
query_file = "challenge_data/train-00000-of-00001.parquet"
df_passages = pd.read_parquet(passage_file)
df_queries = pd.read_parquet(query_file)

# 仅处理前 2420 个 query
df_queries = df_queries.iloc[0:2420]

# 图片目录
query_image_dir = "query_images/"
passage_image_dir = "passage_images/Challenge"

# ----------------------------------------------------
# 4. 预处理函数
# ----------------------------------------------------
def clean_text(text):
    """清理文本，去除非 ASCII 字符以及多余的空格。"""
    if not isinstance(text, str):
        return "[EMPTY]"
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "[EMPTY]"

def check_image_valid(image_path, base_dir):
    """检查图片是否存在且能正常读取，若正常则返回完整路径，否则返回 None。"""
    full_path = os.path.join(base_dir, image_path)
    if os.path.exists(full_path):
        try:
            # 仅做一次读取验证
            with Image.open(full_path) as img:
                img.verify()
            return full_path
        except Exception as e:
            print(f"[ERROR] 读取图片失败: {full_path}, 错误: {e}")
    return None

# ----------------------------------------------------
# 5. 统一预处理 Passage（文本+图片），并计算向量
# ----------------------------------------------------
print("[INFO] Preprocessing and embedding Passages...")

df_passages["cleaned_content"] = df_passages["passage_content"].apply(clean_text)
df_passages["valid_image"] = df_passages["page_screenshot"].apply(
    lambda x: check_image_valid(x, passage_image_dir)
)

# 按批处理 Passage 文本和图片，转换为向量并存储
passage_embs_list = []
passage_ids = df_passages["passage_id"].tolist()

batch_size = 1  # 可根据显存情况调大
num_passages = len(df_passages)
num_batches_p = (num_passages // batch_size) + (1 if num_passages % batch_size else 0)

with torch.no_grad():
    model.set_processor(MODEL_NAME)
    model.processor.patch_size = 14

    for i in range(num_batches_p):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_passages)

        batch_texts = df_passages["cleaned_content"][start_idx:end_idx].tolist()
        batch_images = df_passages["valid_image"][start_idx:end_idx].tolist()

        # data_process
        candidate_inputs = model.data_process(
            text=batch_texts,
            images=batch_images,
            q_or_c="c"  # 'c' for candidate passages
        )
        # 前向计算 & 取倒数第一层向量
        batch_candi_embs = model(**candidate_inputs, output_hidden_states=True)[:, -1, :]
        batch_candi_embs = torch.nn.functional.normalize(batch_candi_embs, dim=-1)

        # 放在 GPU 上进行计算，但为了防止内存泄漏，可以将最终结果搬回 CPU
        passage_embs_list.append(batch_candi_embs.cpu())

# 将所有 Passage 向量拼接
passage_embs = torch.cat(passage_embs_list, dim=0)  # [num_passages, emb_dim]
print(f"[INFO] Passage embedding shape: {passage_embs.shape}")

del passage_embs_list  # 释放临时列表

# ----------------------------------------------------
# 6. 统一预处理 Query（文本+图片），并计算向量
# ----------------------------------------------------
print("[INFO] Preprocessing and embedding Queries...")

df_queries["combined_query"] = df_queries.apply(
    lambda row: row["instruction"] if pd.isna(row["question"]) else row["instruction"] + " " + row["question"],
    axis=1
)
df_queries["valid_image"] = df_queries["img_path"].apply(
    lambda x: check_image_valid(x, query_image_dir) if pd.notna(x) else None
)

query_ids = df_queries["question_id"].tolist()
num_queries = len(df_queries)
batch_size_q = 1  # Query也可批量处理，根据显存选择合适大小
num_batches_q = (num_queries // batch_size_q) + (1 if num_queries % batch_size_q else 0)

query_embs_list = []

with torch.no_grad():
    for i in range(num_batches_q):
        start_idx = i * batch_size_q
        end_idx = min((i + 1) * batch_size_q, num_queries)

        batch_texts = df_queries["combined_query"][start_idx:end_idx].tolist()
        batch_images = df_queries["valid_image"][start_idx:end_idx].tolist()

        query_inputs = model.data_process(
            text=batch_texts,
            images=batch_images,
            q_or_c="q",  # 'q' for query
            task_instruction=(
                "Retrieve the most relevant document page based on text and image matching."
            )
        )
        batch_query_embs = model(**query_inputs, output_hidden_states=True)[:, -1, :]
        batch_query_embs = torch.nn.functional.normalize(batch_query_embs, dim=-1)

        query_embs_list.append(batch_query_embs.cpu())

query_embs = torch.cat(query_embs_list, dim=0)  # [num_queries, emb_dim]
print(f"[INFO] Query embedding shape: {query_embs.shape}")

del query_embs_list

# ----------------------------------------------------
# 7. 相似度计算并取 Top K
# ----------------------------------------------------
# 将 Passage 向量重新搬回 GPU（如果显存足够，也可将 Query 向量放 GPU，一次性矩阵相乘）
passage_embs_gpu = passage_embs.to(device)
query_embs_gpu = query_embs.to(device)

print("[INFO] Computing similarities and retrieving top K passages...")
top_k = 5

# (num_queries, emb_dim) x (emb_dim, num_passages) -> (num_queries, num_passages)
similarity_matrix = torch.matmul(query_embs_gpu, passage_embs_gpu.T)  # [Q, P]

# 拿到 TopK index
top_scores, top_indices = torch.topk(similarity_matrix, top_k, dim=1)

top_indices = top_indices.cpu().numpy()  # 转回 CPU 以便后续处理
top_scores = top_scores.cpu().numpy()

# ----------------------------------------------------
# 8. 保存结果到 CSV
# ----------------------------------------------------
output_file = "submission_bge_m2kr_part1_2420.csv"
with open(output_file, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["question_id", "passage_id"])

    for i, qid in enumerate(query_ids):
        # top_indices[i] 是该 Query 取到的最相似的 Passage 索引
        best_passage_ids = [passage_ids[idx] for idx in top_indices[i]]
        writer.writerow([qid, str(best_passage_ids)])

print(f"✅ 处理完成，查询 1 到 {len(df_queries)} 结果已保存到 {output_file}")
