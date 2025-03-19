import torch
import pandas as pd
import os
import re
from PIL import Image
from transformers import AutoModel
import csv
import warnings
import ast
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# ----------------------------------------------------
# 1. 设备选择
# ----------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ----------------------------------------------------
# 2. 读取 Passage 向量（避免重复计算）
# ----------------------------------------------------
passage_emb_data = torch.load("emb/passage_merged.pt")
all_passage_ids = passage_emb_data["passage_ids"]       # 应为列表
all_passage_embs = passage_emb_data["passage_embs"]       # Tensor [num_passages, emb_dim]
all_passage_embs = all_passage_embs.to(device)

print(f"[INFO] Loaded {len(all_passage_ids)} passages from passage_merged.pt")
print(f"[INFO] Passage embedding shape: {all_passage_embs.shape}")

# 构建 {passage_id: index_in_tensor} 字典
passage_id_to_idx = {pid: idx for idx, pid in enumerate(all_passage_ids)}

# ----------------------------------------------------
# 3. 定义模型（仅用于计算 Query 向量）
# ----------------------------------------------------
MODEL_NAME = "BAAI/BGE-VL-MLLM-S2"
print(f"[INFO] Loading model for query embedding: {MODEL_NAME}")
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, device_map="auto")
model.eval()

# ----------------------------------------------------
# 4. 加载数据（Query & Candidates）
# ----------------------------------------------------
query_file = "challenge_data/train-00000-of-00001.parquet"
candidate_file = "output/modified_submission-m2kr-top100.csv"

df_queries = pd.read_parquet(query_file)
df_candidates = pd.read_csv(candidate_file)

# 若需要处理全部 query，这里直接使用整个 df_queries
df_queries = df_queries

# 将候选 passage 列由字符串转换为列表
df_candidates["passage_id"] = df_candidates["passage_id"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
# 将候选文件中的 question_id 也转换为字符串，确保和 df_queries 匹配
df_candidates["question_id"] = df_candidates["question_id"].astype(str)

query_image_dir = "query_images/"

def clean_text(text):
    if not isinstance(text, str):
        return "[EMPTY]"
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "[EMPTY]"

def check_image_valid(image_path, base_dir):
    if pd.isna(image_path):
        return None
    full_path = os.path.join(base_dir, image_path)
    if os.path.exists(full_path):
        try:
            with Image.open(full_path) as img:
                img.verify()
            return full_path
        except Exception:
            pass
    return None

# 调试输出 df_queries 的列和部分内容
print("df_queries columns:", df_queries.columns)
print("df_queries head:\n", df_queries.head())

# 强制将 df_queries["question_id"] 转换为字符串，避免出现内置函数 id 问题
df_queries["question_id"] = df_queries["question_id"].astype(str)

# ----------------------------------------------------
# 5. 对 Query 进行小批量处理，每批 2 条，然后立即 Rerank & 输出
# ----------------------------------------------------
print("[INFO] Processing queries in batches of size 2, then Rerank...")

df_queries["combined_query"] = df_queries.apply(
    lambda row: row["instruction"] if pd.isna(row["question"])
                else row["instruction"] + " " + row["question"],
    axis=1
)
df_queries["valid_image"] = df_queries["img_path"].apply(
    lambda x: check_image_valid(x, query_image_dir)
)

# 提取 query id 列（转换为字符串后）
query_ids = df_queries["question_id"].tolist()
num_queries = len(df_queries)
print("Sample query ids:", query_ids[:10])

model.set_processor(MODEL_NAME)
model.processor.patch_size = 14

output_file = "submission_bge_r_rank_top5.csv"
top_k = 5

batch_size_q = 12
num_batches_q = (num_queries + batch_size_q - 1) // batch_size_q

with open(output_file, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["question_id", "passage_id"])
    f.flush()

    # 用 tqdm 包装批次循环
    for batch_idx in tqdm(range(num_batches_q), desc="Compute & Rerank"):

        start_i = batch_idx * batch_size_q
        end_i = min((batch_idx + 1) * batch_size_q, num_queries)

        # 取出这一小批的 query 内容
        batch_texts = df_queries["combined_query"][start_i:end_i].tolist()
        batch_images = df_queries["valid_image"][start_i:end_i].tolist()
        batch_ids = query_ids[start_i:end_i]  # Query ID 列表

        # ====== 1. 计算这个小批量的 Query 向量 ======
        with torch.no_grad():
            query_inputs = model.data_process(
                text=batch_texts,
                images=batch_images,
                q_or_c="q",
                task_instruction="Retrieve the most relevant document page based on text and image matching."
            )
            batch_query_embs = model(**query_inputs, output_hidden_states=True)[:, -1, :]
            batch_query_embs = torch.nn.functional.normalize(batch_query_embs, dim=-1)
            batch_query_embs = batch_query_embs.to(device)  # shape [batch_size_q, emb_dim]

        # ====== 2. 对此小批量里的每个 Query 做 Rerank & 输出 ======
        for i_in_batch, qid in enumerate(batch_ids):
            # 调试输出当前处理的 query id
            #print(f"[INFO] Processing Query ID: {qid}")

            candidate_rows = df_candidates.loc[df_candidates["question_id"] == qid, "passage_id"].values
            if len(candidate_rows) == 0:
                print(f"[WARNING] No candidate passages found for query id: {qid}")
                continue
            candidate_passage_ids = candidate_rows[0]
            candidate_passage_ids = candidate_passage_ids[:50]

            # 在 all_passage_embs 中找到这些向量
            candidate_indices = [passage_id_to_idx[pid] for pid in candidate_passage_ids if pid in passage_id_to_idx]
            if len(candidate_indices) == 0:
                print(f"[WARNING] None of the candidate passage IDs for query id {qid} are in passage embeddings.")
                continue

            candidate_vectors = all_passage_embs[candidate_indices]  # shape [n_candidates, emb_dim]

            # 当前 Query 的向量
            query_vector = batch_query_embs[i_in_batch].unsqueeze(0)  # [1, emb_dim]

            # 相似度计算（点乘）
            similarities = torch.matmul(query_vector, candidate_vectors.T).squeeze(0)
            top_indices = torch.topk(similarities, top_k).indices.cpu().numpy()
            best_passages = [candidate_passage_ids[idx] for idx in top_indices]

            writer.writerow([qid, str(best_passages)])
            f.flush()  # 每处理完一个 query 就 flush

            #print(f"[INFO] Finished Query ID: {qid}")

print(f"✅ 处理完成，结果已保存到 {output_file}")
