import os
import re
import csv
import json
import torch
import pandas as pd
from PIL import Image
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
    # 老版本不支持 init_device，所以这里省略
)

# ---------- 2. 设置 tokenizer，避免右侧填充的警告 ----------
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# ---------- 3. 加载 OpenFlamingo-9B 预训练权重 ----------
checkpoint_path = hf_hub_download(
    repo_id="openflamingo/OpenFlamingo-9B-vitl-mpt7b",
    filename="checkpoint.pt"
)
model.load_state_dict(torch.load(checkpoint_path), strict=False)
model.to(device)
model.lang_encoder.config.pad_token_id = tokenizer.eos_token_id

print("✅ OpenFlamingo-9B 预训练模型加载完成！")

# ---------- 4. 读取文档库（`challenge_passage.parquet`） ----------
df_passages = pd.read_parquet("challenge_passage/train-00000-of-00001.parquet")

# ---------- 5. 读取查询数据（`challenge_data.parquet`） ----------
df_queries = pd.read_parquet("challenge_data/train-00000-of-00001.parquet")

# ---------- 6. 图片文件目录 ----------
image_dir = "query_images/"  # 确保这个目录包含所有 `img_path` 指定的图片

# ---------- 7. 检索函数：从一条 query 中检索前 k 条相关的 Passage ----------
def retrieve_top_k_passages(question_id, instruction, question, img_path, top_k=5):
    # 先把 instruction 当作基础 Query
    query_text = instruction

    # ★★★始终准备一个 dummy_vision_x 用于无图像时的生成
    dummy_vision_x = torch.zeros(1, 1, 1, 3, 224, 224).to(device)

    # 如果有图片，则使用图像 + OpenFlamingo 生成一些查询扩展文本
    if pd.notna(img_path):
        image_path = os.path.join(image_dir, img_path)
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")

            # ★ 3 次 unsqueeze 以满足 (b, T_img, F, C, H, W) = 6D
            image_tensor = image_processor(image)  # (3, H, W)
            image_tensor = image_tensor.unsqueeze(0)  # (1, 3, H, W)
            image_tensor = image_tensor.unsqueeze(1)  # (1, 1, 3, H, W)
            image_tensor = image_tensor.unsqueeze(2)  # (1, 1, 1, 3, H, W)
            image_tensor = image_tensor.to(device)

            # 准备文本输入
            lang_x = tokenizer(["<image>" + instruction + "<|endofchunk|>"], return_tensors="pt",truncation=True,
                           max_length=2048)
            input_ids = lang_x["input_ids"].to(device)
            attention_mask = lang_x["attention_mask"].to(device)

            # 用模型生成图像描述文本
            with torch.no_grad():
                output = model.generate(
                    vision_x=image_tensor,  # 这里可以传真图像
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=10

                )

            generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            query_text += " " + generated_text
        else:
            print(f"⚠️ 图片文件 {image_path} 不存在，跳过")
    else:
        # 没有图片则什么都不做，后续照旧
        pass

    # 如果有 question，则拼接上 question
    if pd.notna(question):
        query_text += " " + question

    # 用 OpenFlamingo 算每个 Passage 的相关度分数
    scores = []
    for _, passage in df_passages.iterrows():
        passage_text = passage["passage_content"]

        # 给大模型一个 prompt，要求它给出相关性评分
        input_text = f"Query: {query_text}\nPassage: {passage_text}\n相关性评分 (0-1):"

        lang_x = tokenizer(["<image>" + instruction + "<|endofchunk|>"], return_tensors="pt", truncation=True,
                           max_length=2048)
        input_ids = lang_x["input_ids"].to(device)
        attention_mask = lang_x["attention_mask"].to(device)

        with torch.no_grad():
            # ★ 这里一定也要传 vision_x=dummy_vision_x 而不是 None
            output = model.generate(
                vision_x=dummy_vision_x,
                lang_x=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10
            )

        generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        # 用正则表达式从生成文本里解析浮点数评分
        match = re.search(r"(\d+\.\d+)", generated_text)
        if match:
            score = float(match.group(1))
        else:
            score = 0  # 如果解析不到，就设成 0

        scores.append((score, passage["passage_id"]))

    # 按照分数从高到低排序，取 top_k 个 passage
    sorted_passages = sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
    return [passage_id for _, passage_id in sorted_passages]

# ---------- 8. 生成结果文件 `submission.csv` ----------
submission_file = "submission_m2kr.csv"
with open(submission_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question_id", "passage_id"])

    for _, query in df_queries.iterrows():
        question_id = query["question_id"]
        instruction = query["instruction"]
        question = query["question"] if pd.notna(query["question"]) else None
        img_path = query["img_path"] if pd.notna(query["img_path"]) else None

        top_passages = retrieve_top_k_passages(question_id, instruction, question, img_path, top_k=5)

        for passage_id in top_passages:
            writer.writerow([question_id, passage_id])

print(f"✅ Submission file generated: {submission_file}")
