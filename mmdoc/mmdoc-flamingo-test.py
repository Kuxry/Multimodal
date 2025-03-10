import os
import re
import csv
import json
import io
import torch
import pandas as pd
from PIL import Image

# ========= OpenFlamingo 相关 ========
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms

# ---------- 设备选择 ----------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ---------- 1. 创建模型、图像处理器和分词器 ----------
model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-7b",
    tokenizer_path="anas-awadalla/mpt-7b",
    cross_attn_every_n_layers=4,
)

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# ---------- 2. 加载 OpenFlamingo 预训练权重 ----------
checkpoint_path = hf_hub_download(
    repo_id="openflamingo/OpenFlamingo-9B-vitl-mpt7b",
    filename="checkpoint.pt"
)
model.load_state_dict(torch.load(checkpoint_path), strict=False)
model.to(device)
# 给 MPT 模型设置 pad_token_id = eos_token_id，避免警告
model.lang_encoder.config.pad_token_id = tokenizer.eos_token_id

print("✅ OpenFlamingo-9B 模型加载完成！")

# ---------- 3. 读取文档库（MMDocIR_doc_passages.parquet） ----------
df_passages = pd.read_parquet("MMDocIR_doc_passages.parquet")

# ---------- 4. 读取测试集查询（MMDocIR_gt_remove.jsonl） ----------
queries = []
with open("MMDocIR_gt_remove.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        queries.append(json.loads(line.strip()))

# ---------- 5. 利用 OpenFlamingo 做相似度打分的函数 ----------
def retrieve_top_k_passages(question, doc_name, top_k=5):
    """
    question: 用户的文本问题
    doc_name: 当前文档名称
    top_k: 取前 k 个相关度最高的passage

    返回: 排好序的 passage_id 列表
    """

    # 过滤出该文档的所有页面
    doc_pages = df_passages[df_passages["doc_name"] == doc_name]

    scores_and_ids = []

    # 先为“无图像场景”准备一个 dummy 张量，保证 shape=(1,1,1,3,224,224)
    dummy_vision_x = torch.zeros(1, 1, 1, 3, 224, 224, device=device)

    for _, passage in doc_pages.iterrows():
        passage_id = passage["passage_id"]

        # 读取 VLM、OCR 字段（若无则为空字符串）
        vlm_text = passage.get("vlm_text", "").strip()
        ocr_text = passage.get("ocr_text", "").strip()

        # 判断优先使用 VLM 或者 OCR 的文本
        if vlm_text:
            passage_content = vlm_text
            vision_x = dummy_vision_x  # 占位图像
        elif ocr_text:
            passage_content = ocr_text
            vision_x = dummy_vision_x
        else:
            passage_content = None
            # 把二进制图像转换为 PIL
            image_data = passage["image_binary"]
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            # 处理成 OpenFlamingo 需要的 6D 张量
            image_tensor = image_processor(image).unsqueeze(0).unsqueeze(1).unsqueeze(2).to(device)
            vision_x = image_tensor

        # **减少拼接**
        if passage_content is not None:
            prompt_for_gen = f"{passage_content[:500]}\n{question[:500]}<|endofchunk|>"
        else:
            prompt_for_gen = f"<image>{question[:500]}<|endofchunk|>"

        img_lang_x = tokenizer(
            [prompt_for_gen],
            return_tensors="pt",
            truncation=True,
            max_length=1024  # 预留空间给生成的 token
        )

        with torch.no_grad():
            out_img = model.generate(
                vision_x=vision_x,
                lang_x=img_lang_x["input_ids"].to(device),
                attention_mask=img_lang_x["attention_mask"].to(device),
                max_new_tokens=10  # 限制生成 token 数量
            )
        # **避免无限拼接**
        generated_text = tokenizer.batch_decode(out_img, skip_special_tokens=True)[0][:500]
        enriched_question = f"{question[:500]} {generated_text}"

        # **控制最终 `input_text` 长度**
        if passage_content is not None:
            input_text = (
                f"Query: {enriched_question[:500]}\n"
                f"Passage: {passage_content[:1200]}\n"
                f"相关性评分 (0-1):"
            )
        else:
            input_text = (
                f"Query: {enriched_question[:500]}\n"
                f"Passage: <image>\n"
                f"相关性评分 (0-1):"
            )

        score_lang_x = tokenizer(
            [input_text],
            return_tensors="pt",
            truncation=True,
            max_length=1024  # 确保不会超出 2048
        )

        with torch.no_grad():
            out_score = model.generate(
                vision_x=vision_x,
                lang_x=score_lang_x["input_ids"].to(device),
                attention_mask=score_lang_x["attention_mask"].to(device),
                max_new_tokens=5  # 限制最终生成 token
            )

        score_text = tokenizer.batch_decode(out_score, skip_special_tokens=True)[0]
        match = re.search(r"(\d+\.\d+)", score_text)
        score_val = float(match.group(1)) if match else 0.0

        scores_and_ids.append((score_val, passage_id))

    sorted_by_score = sorted(scores_and_ids, key=lambda x: x[0], reverse=True)[:top_k]
    best_passage_ids = [pid for _, pid in sorted_by_score]
    return best_passage_ids

# ---------- 6. 生成结果文件 submission_mmdoc.csv ----------
submission_file = "submission_mmdoc_1.csv"
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
