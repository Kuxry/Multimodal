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
device = "cuda" if torch.cuda.is_available() else "cpu"

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

# ---------- 3. 读取文档库（`MMDocIR_doc_passages.parquet`） ----------
#    里边包含多行，每行有 doc_name、passage_id、image_binary 等信息
df_passages = pd.read_parquet("MMDocIR_doc_passages.parquet")

# ---------- 4. 读取测试集查询（`MMDocIR_gt_remove.jsonl`） ----------
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

    # 遍历该 doc_name 下的所有 passage
    for _, passage in doc_pages.iterrows():
        passage_id = passage["passage_id"]

        # 把二进制图像转换为 PIL
        image_data = passage["image_binary"]
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # 把 PIL 图像变成 OpenFlamingo 需要的 6D 张量
        image_tensor = image_processor(image)           # shape: (3, H, W)
        image_tensor = image_tensor.unsqueeze(0)        # (1, 3, H, W)
        image_tensor = image_tensor.unsqueeze(1)        # (1, 1, 3, H, W)
        image_tensor = image_tensor.unsqueeze(2)        # (1, 1, 1, 3, H, W)
        image_tensor = image_tensor.to(device)

        # ========== 先让模型对图片 + question 做一次生成，得到更多上下文描述 ==========
        # 拼一个 prompt，例如 "<image> + question + <|endofchunk|>"
        prompt_for_image = f"<image>{question}<|endofchunk|>"
        img_lang_x = tokenizer(
            [prompt_for_image],
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        with torch.no_grad():
            out_img = model.generate(
                vision_x=image_tensor,
                lang_x=img_lang_x["input_ids"].to(device),
                attention_mask=img_lang_x["attention_mask"].to(device),
                max_new_tokens=20
            )
        # 把生成文本解码并附加到 question 上，形成更丰富的 query
        generated_text = tokenizer.batch_decode(out_img, skip_special_tokens=True)[0]
        enriched_question = question + " " + generated_text

        # ========== 然后让模型去对 "enriched_question" 和 "passage" 做相关性打分 ==========
        #   prompt 里让它直接输出一个 [0-1] 评分
        input_text = (
            f"Query: {enriched_question}\n"
            f"Passage: (the passage image is above)\n"
            f"相关性评分 (0-1):"
        )
        # 注意：可以把 “passage_content” 也拼进去，如果有的话
        # 若你只想用图像进行评分，就写 “Passage: <image>” + ...
        # 这里演示仅写一句说明

        score_lang_x = tokenizer(
            [input_text],
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        with torch.no_grad():
            out_score = model.generate(
                vision_x=image_tensor,   # 这次继续把图像信息传入
                lang_x=score_lang_x["input_ids"].to(device),
                attention_mask=score_lang_x["attention_mask"].to(device),
                max_new_tokens=10
            )

        score_text = tokenizer.batch_decode(out_score, skip_special_tokens=True)[0]
        # 用正则提取可能的浮点数
        match = re.search(r"(\d+\.\d+)", score_text)
        if match:
            score_val = float(match.group(1))
        else:
            score_val = 0.0

        scores_and_ids.append((score_val, passage_id))

    # 按分数从高到低排序，取前 top_k
    sorted_by_score = sorted(scores_and_ids, key=lambda x: x[0], reverse=True)[:top_k]
    best_passage_ids = [pid for _, pid in sorted_by_score]
    return best_passage_ids

# ---------- 6. 生成结果文件 `submission_mmdoc.csv` ----------
submission_file = "submission_mmdoc.csv"
with open(submission_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question_id", "passage_id"])

    for item in queries:
        question_id = item["question_id"]
        doc_name = item["doc_name"]
        # question 内容
        question = item["question"]  # 例如 "请解释文档里某段内容..."

        # 调用检索函数
        top_passages = retrieve_top_k_passages(question, doc_name, top_k=5)

        # 把 top_passages 写到 CSV
        # 注意：你想按行写多个ID，还是只写一个，都可自行决定
        # 下面演示把list直接转字符串写进去：
        writer.writerow([question_id, str(top_passages)])

print(f"✅ Submission file generated: {submission_file}")
