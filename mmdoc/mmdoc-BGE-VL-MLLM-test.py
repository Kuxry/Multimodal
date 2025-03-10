import os
import re
import csv
import json
import io
import torch
import pandas as pd
from PIL import Image
from transformers import AutoModel

# ========== 加载 BAAI/BGE-VL-MLLM-S2 模型 ==========
MODEL_NAME = "BAAI/BGE-VL-MLLM-S2"

model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
model.cuda()

# 设置处理器（processor）
with torch.no_grad():
    model.set_processor(MODEL_NAME)

print("✅ BAAI/BGE-VL-MLLM-S2 模型加载完成！")

# ========== 读取文档库（MMDocIR_doc_passages.parquet） ==========
df_passages = pd.read_parquet("MMDocIR_doc_passages.parquet")

# ========== 读取测试集查询（MMDocIR_gt_remove.jsonl） ==========
queries = []
with open("MMDocIR_gt_remove.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        queries.append(json.loads(line.strip()))

# ========== 利用 BAAI 模型做相似度打分的函数 ==========
def retrieve_top_k_passages(question, doc_name, top_k=5):
    """
    question: 用户的文本问题
    doc_name: 当前文档名称
    top_k: 返回前 k 个相关度最高的 passage

    返回: 排序后的 passage_id 列表
    """
    # 过滤出该文档的所有页面
    doc_pages = df_passages[df_passages["doc_name"] == doc_name]
    scores_and_ids = []

    # 计算 query 的 embedding
    with torch.no_grad():
        query_input = model.data_process(
            text=question,
            q_or_c="q"
        )
        query_emb = model(**query_input, output_hidden_states=True)[:, -1, :]
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

    for _, passage in doc_pages.iterrows():
        passage_id = passage["passage_id"]

        # 读取 VLM、OCR 字段（若无则为空字符串）
        vlm_text = passage.get("vlm_text", "").strip()
        ocr_text = passage.get("ocr_text", "").strip()

        # 根据 passage 是否有文本内容决定 candidate 的输入形式
        if vlm_text:
            candidate_input = model.data_process(
                text=vlm_text,
                q_or_c="c"
            )
        elif ocr_text:
            candidate_input = model.data_process(
                text=ocr_text,
                q_or_c="c"
            )
        else:
            # 如果没有文本，则使用图像
            image_data = passage["image_binary"]
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            candidate_input = model.data_process(
                images=image,
                q_or_c="c"
            )

        with torch.no_grad():
            candidate_emb = model(**candidate_input, output_hidden_states=True)[:, -1, :]
            candidate_emb = torch.nn.functional.normalize(candidate_emb, dim=-1)

        # 计算余弦相似度（点积）
        score = torch.matmul(query_emb, candidate_emb.T)
        score_val = score.item()
        scores_and_ids.append((score_val, passage_id))

    # 取相似度最高的 top_k 个 passage_id
    sorted_by_score = sorted(scores_and_ids, key=lambda x: x[0], reverse=True)[:top_k]
    best_passage_ids = [pid for _, pid in sorted_by_score]
    return best_passage_ids

# ========== 生成结果文件 submission_mmdoc.csv ==========
submission_file = "submission_mmdoc.csv"
with open(submission_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question_id", "passage_id"])

    for item in queries:
        question_id = item["question_id"]
        doc_name = item["doc_name"]
        question = item["question"]

        top_passages = retrieve_top_k_passages(question, doc_name, top_k=5)

        writer.writerow([question_id, str(top_passages)])

print(f"✅ Submission file generated: {submission_file}")
