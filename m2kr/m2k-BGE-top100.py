import torch
import pandas as pd
import os
import re
from PIL import Image
from transformers import AutoModel
import csv
import ast
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
# 3. 加载数据（Passage & Query & Candidates）
# ----------------------------------------------------
passage_file = "challenge_passage/train-00000-of-00001.parquet"
query_file = "challenge_data/train-00000-of-00001.parquet"
candidate_file = "output/retrieval_results-top100.csv"  # 你的输入文件

df_passages = pd.read_parquet(passage_file)
df_queries = pd.read_parquet(query_file)
df_candidates = pd.read_csv(candidate_file)

# 仅处理前 2420 个 query
df_queries = df_queries.iloc[0:2420]
df_candidates = df_candidates[df_candidates["query_id"].isin(df_queries["question_id"])]

# 字段映射
df_candidates.rename(columns={"query_id": "question_id"}, inplace=True)

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

# 解析 top_passage_ids 字符串为列表
df_candidates["top_passage_ids"] = df_candidates["top_passage_ids"].apply(ast.literal_eval)

# ----------------------------------------------------
# 5. 仅计算候选 Passage 的向量
# ----------------------------------------------------
print("[INFO] Filtering candidate passages and computing embeddings...")

# 取出所有候选 passage_id 并去重
candidate_passage_ids = set(df_candidates["top_passage_ids"].explode())
df_passages = df_passages[df_passages["passage_id"].isin(candidate_passage_ids)]

df_passages["cleaned_content"] = df_passages["passage_content"].apply(clean_text)

# 计算候选 passage 的向量
passage_embs_list = []
passage_ids = df_passages["passage_id"].tolist()

batch_size = 1  # 依据显存情况调整
num_passages = len(df_passages)
num_batches_p = (num_passages // batch_size) + (1 if num_passages % batch_size else 0)

with torch.no_grad():
    model.set_processor(MODEL_NAME)
    model.processor.patch_size = 14

    for i in range(num_batches_p):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_passages)

        batch_texts = df_passages["cleaned_content"][start_idx:end_idx].tolist()

        candidate_inputs = model.data_process(
            text=batch_texts,
            images=None,
            q_or_c="c"
        )
        batch_candi_embs = model(**candidate_inputs, output_hidden_states=True)[:, -1, :]
        batch_candi_embs = torch.nn.functional.normalize(batch_candi_embs, dim=-1)

        passage_embs_list.append(batch_candi_embs.cpu())

passage_embs = torch.cat(passage_embs_list, dim=0)
print(f"[INFO] Passage embedding shape: {passage_embs.shape}")

del passage_embs_list  # 释放内存

# ----------------------------------------------------
# 6. 计算 Query 向量
# ----------------------------------------------------
print("[INFO] Computing query embeddings...")

df_queries["combined_query"] = df_queries.apply(
    lambda row: row["instruction"] if pd.isna(row["question"]) else row["instruction"] + " " + row["question"],
    axis=1
)

query_ids = df_queries["question_id"].tolist()
num_queries = len(df_queries)
batch_size_q = 1
num_batches_q = (num_queries // batch_size_q) + (1 if num_queries % batch_size_q else 0)

query_embs_list = []

with torch.no_grad():
    for i in range(num_batches_q):
        start_idx = i * batch_size_q
        end_idx = min((i + 1) * batch_size_q, num_queries)

        batch_texts = df_queries["combined_query"][start_idx:end_idx].tolist()

        query_inputs = model.data_process(
            text=batch_texts,
            images=None,
            q_or_c="q",
            task_instruction="Retrieve the most relevant document page based on text matching."
        )
        batch_query_embs = model(**query_inputs, output_hidden_states=True)[:, -1, :]
        batch_query_embs = torch.nn.functional.normalize(batch_query_embs, dim=-1)

        query_embs_list.append(batch_query_embs.cpu())

query_embs = torch.cat(query_embs_list, dim=0)
print(f"[INFO] Query embedding shape: {query_embs.shape}")

del query_embs_list  # 释放内存

# ----------------------------------------------------
# 7. 计算候选 Passage 内的相似度，取 Top 5
# ----------------------------------------------------
print("[INFO] Retrieving top 5 passages from the 100 candidates...")

passage_embs_gpu = passage_embs.to(device)
query_embs_gpu = query_embs.to(device)

top_k = 5
output_file = "submission_bge_r_rank_top5.csv"

with open(output_file, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["question_id", "passage_id"])

    for qid in query_ids:
        candidate_passage_ids = df_candidates[df_candidates["question_id"] == qid]["top_passage_ids"].values[0]

        # 过滤候选 Passage 向量
        candidate_indices = [passage_ids.index(pid) for pid in candidate_passage_ids]
        candidate_vectors = passage_embs_gpu[candidate_indices]

        # 计算相似度
        query_vector = query_embs_gpu[query_ids.index(qid)].unsqueeze(0)
        similarities = torch.matmul(query_vector, candidate_vectors.T).squeeze(0)

        # 取 Top 5
        top_indices = torch.topk(similarities, top_k).indices.cpu().numpy()
        best_passages = [candidate_passage_ids[idx] for idx in top_indices]

        writer.writerow([qid, str(best_passages)])

print(f"✅ 处理完成，结果已保存到 {output_file}")
