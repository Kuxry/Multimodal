import pandas as pd
import json
import torch
import csv
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import Blip2Processor, BlipForConditionalGeneration

# ✅ 设备选择
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 加载 BLIP2 模型
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", ignore_mismatched_sizes=True).to(device)

# ✅ 读取 `challenge_passage.parquet`（文档库）
df_passages = pd.read_parquet("challenge_passage/train-00000-of-00001.parquet")

# ✅ 读取 `challenge_data.parquet`（查询数据）
df_queries = pd.read_parquet("challenge_data/train-00000-of-00001.parquet")

# ✅ 图片文件目录
image_dir = "images/"  # 确保这个目录包含所有 `img_path` 指定的图片

# ✅ `query` - `passage` 检索匹配函数
def retrieve_top_k_passages(question_id, instruction, question, img_path, top_k=5):
    query_text = instruction  # 任务描述作为基础文本
    scores, passage_ids = [], []

    # 处理查询图片（如果存在）
    if pd.notna(img_path):
        image_path = os.path.join(image_dir, img_path)  # 组合完整路径
        if os.path.exists(image_path):  # 确保图片存在
            image = Image.open(image_path).convert("RGB")

            # 处理图片并生成文本描述
            inputs = processor(image, instruction, return_tensors="pt").to(device)
            output = model.generate(**inputs)
            generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]

            query_text += " " + generated_text  # 追加 BLIP2 生成的文本描述
        else:
            print(f"⚠️ 图片文件 {image_path} 不存在，跳过")

    # 处理文本查询（如果存在）
    if pd.notna(question):
        query_text += " " + question  # 追加文本查询内容

    # 计算 `query` 与所有 `passage_content` 的相似度
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([query_text] + df_passages["passage_content"].tolist())
    query_vector = tfidf_matrix[0]  # 查询向量
    passage_vectors = tfidf_matrix[1:]  # 文档向量

    similarities = cosine_similarity(query_vector, passage_vectors).flatten()
    sorted_indices = similarities.argsort()[::-1][:top_k]  # 取 top-k 最高分的索引

    return df_passages.iloc[sorted_indices]["passage_id"].tolist()

# ✅ 生成 `submission.csv`
submission_file = "submission_m2kr.csv"
with open(submission_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["question_id", "passage_id"])

    for _, query in df_queries.iterrows():
        question_id = query["question_id"]
        instruction = query["instruction"]
        question = query["question"] if pd.notna(query["question"]) else None
        img_path = query["img_path"] if pd.notna(query["img_path"]) else None

        top_passages = retrieve_top_k_passages(question_id, instruction, question, img_path, top_k=5)

        writer.writerow([question_id, str(top_passages)])

print(f"✅ Submission file generated: {submission_file}")
