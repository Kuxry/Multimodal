import pandas as pd
import json
import torch
import csv
import io
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import Blip2Processor
from transformers import BlipForConditionalGeneration

# ✅ 设备选择
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ✅ 让模型加载到正确的设备
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", ignore_mismatched_sizes=True).to(device)

# ✅ 读取 `MMDocIR_doc_passages.parquet`
df = pd.read_parquet("mmdoc/MMDocIR_doc_passages.parquet")

# ✅ 读取 `MMDocIR_gt_remove.jsonl`（测试集查询）
queries = []
with open("mmdoc/MMDocIR_gt_remove.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        queries.append(json.loads(line.strip()))

# ✅ `query` - `passage` 匹配函数
def retrieve_top_k_passages(query, doc_name, top_k=5):
    doc_pages = df[df["doc_name"] == doc_name]  # 获取该文档的所有页面
    scores, passage_ids = [], []

    for _, passage in doc_pages.iterrows():
        image = Image.open(io.BytesIO(passage["image_binary"])).convert("RGB")

        # 处理图片并生成文本描述
        inputs = processor(image, query, return_tensors="pt").to(device)  # ✅ 确保 inputs 也在 `cuda:1`
        output = model.generate(**inputs)
        generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]

        # 计算 `query` 和 `generated_text` 余弦相似度
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([query, generated_text])
        similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

        scores.append(similarity_score)
        passage_ids.append(passage["passage_id"])

    # 返回 top-k `passage_id`
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [passage_ids[i] for i in sorted_indices[:top_k]]

# ✅ 生成 `submission.csv`
submission_file = "submission_mmdoc.csv"
with open(submission_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["question_id", "passage_id"])

    for item in queries:
        question_id = item["question_id"]
        doc_name = item["doc_name"]

        top_passages = retrieve_top_k_passages(item["question"], doc_name, top_k=5)

        writer.writerow([question_id, str(top_passages)])

print(f"✅ Submission file generated: {submission_file}")
