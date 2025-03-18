import torch
import pandas as pd
import os
import re
from PIL import Image
from transformers import AutoModel
import warnings

# 允许处理超大图片，避免 PIL 抛出 DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# ---------------------------------------------------
# 1. 基本设置
# ---------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

MODEL_NAME = "BAAI/BGE-VL-MLLM-S2"

# Query 数据文件
QUERY_FILE = "challenge_data/train-00000-of-00001.parquet"
# Query 图片所在文件夹
QUERY_IMAGE_DIR = "query_images/"
# 输出：保存前半部分 Query 向量的 .pt 文件
OUTPUT_QUERY_EMB_FILE = "query_embeddings_challenge_data_part1.pt"

# ---------------------------------------------------
# 2. 加载模型
# ---------------------------------------------------
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
model.eval()
model.set_processor(MODEL_NAME)
model.processor.patch_size = 14

# ---------------------------------------------------
# 3. 加载并截取前半部分 Query 数据
# ---------------------------------------------------
df_queries = pd.read_parquet(QUERY_FILE)
total_queries = len(df_queries)
print(f"[INFO] Total queries in file: {total_queries}")

# 截取前 3207 行 (你也可根据 6415 // 2 调整到 3208 等)
df_queries = df_queries.iloc[:3207]
print(f"[INFO] Now processing first {len(df_queries)} queries (0 ~ 3206)")

# ---------------------------------------------------
# 4. 文本 & 图片预处理函数
# ---------------------------------------------------
def clean_text(text):
    """
    简单清洗文本，去除非 ASCII 字符、多余空格等。
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def check_image_valid(image_path, base_dir):
    """
    检查图片是否存在、能否正常读取。
    返回完整路径或 None。
    """
    if not isinstance(image_path, str):
        return None
    full_path = os.path.join(base_dir, image_path)
    if os.path.exists(full_path):
        try:
            with Image.open(full_path) as img:
                img.verify()
            return full_path
        except Exception as e:
            print(f"[ERROR] 读取图片失败: {full_path}, 错误: {e}")
    return None

# ---------------------------------------------------
# 5. 清洗 instruction/question，并组合文本
# ---------------------------------------------------
df_queries["instruction_cleaned"] = df_queries["instruction"].apply(clean_text)
df_queries["question_cleaned"] = df_queries["question"].apply(clean_text)

def combine_query_text(row):
    # 如果 question_cleaned 为空，就只用 instruction_cleaned
    # 否则拼接 "instruction question"
    instr = row["instruction_cleaned"]
    ques = row["question_cleaned"]
    if len(ques) == 0:
        return instr
    else:
        return f"{instr} {ques}"

df_queries["combined_text"] = df_queries.apply(combine_query_text, axis=1)

# ---------------------------------------------------
# 6. 处理图片路径（img_path）
# ---------------------------------------------------
df_queries["valid_image"] = df_queries["img_path"].apply(
    lambda x: check_image_valid(x, QUERY_IMAGE_DIR)
)

# 如果要过滤无效图片，可启用：
# df_queries = df_queries[~df_queries["valid_image"].isnull()].copy()
# print(f"[INFO] After filter, valid queries: {len(df_queries)}")

# 提取 ID、文本、图片，以及 instruction
query_ids = df_queries["question_id"].tolist()
all_texts = df_queries["combined_text"].tolist()
all_images = df_queries["valid_image"].tolist()

# 在此示例中，我们把 instruction_cleaned 当做 task_instruction
all_instructions = df_queries["instruction_cleaned"].tolist()

# ---------------------------------------------------
# 7. 批量生成 Query 向量 (前半部分)
# ---------------------------------------------------
batch_size = 2  # 根据显存自由调整
num_queries = len(df_queries)
num_batches = (num_queries + batch_size - 1) // batch_size

print(f"[INFO] Start embedding first half queries: total {num_queries}, batch_size={batch_size}, total batches={num_batches}")

emb_list = []
with torch.no_grad():
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_queries)

        batch_texts = all_texts[start_idx:end_idx]
        batch_imgs = all_images[start_idx:end_idx]
        batch_instructions = all_instructions[start_idx:end_idx]

        # data_process: q_or_c="q" 表示Query
        query_inputs = model.data_process(
            text=batch_texts,
            images=batch_imgs,
            q_or_c="q",
            task_instruction=batch_instructions
        )

        # 前向计算
        batch_embs = model(**query_inputs, output_hidden_states=True)[:, -1, :]
        batch_embs = torch.nn.functional.normalize(batch_embs, dim=-1)

        # 搬回CPU，释放显存
        emb_list.append(batch_embs.cpu())

        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            print(f"[INFO] Finished batch {i+1}/{num_batches}")

# 拼接所有批次
query_embs = torch.cat(emb_list, dim=0)  # shape: [num_queries, emb_dim]
del emb_list

print(f"[INFO] Completed first-half query embedding. shape = {query_embs.shape}")

# ---------------------------------------------------
# 8. 保存到 .pt 文件
# ---------------------------------------------------
save_data = {
    "query_ids": query_ids,
    "query_embs": query_embs
}
torch.save(save_data, OUTPUT_QUERY_EMB_FILE)

print(f"[INFO] Query embeddings (part1) saved to: {OUTPUT_QUERY_EMB_FILE}")
