import torch
from transformers import AutoModel
from PIL import Image
import pandas as pd
import json
import re
from io import BytesIO


# 过滤无效字符的函数
def clean_text(text):
    if not isinstance(text, str):  # 防御非字符串输入
        return "[EMPTY]"
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "[EMPTY]"


# 读取数据
MMDocIR_gt_file = "MMDocIR_gt_remove.jsonl"
MMDocIR_doc_file = "MMDocIR_doc_passages.parquet"


dataset_df = pd.read_parquet(MMDocIR_doc_file)
data_json = [json.loads(line.strip()) for line in open(MMDocIR_gt_file, 'r', encoding="utf-8")]

# 定义模型名称
MODEL_NAME = "BAAI/BGE-VL-MLLM-S2"

# 加载模型
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
model.eval()


# **创建空白图片**
def create_blank_image():
    img = Image.new("RGB", (64, 64), (255, 255, 255))  # 创建空白白色图片
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)  # 重置指针
    return img_byte_arr  # 返回 BytesIO 对象


# **处理第一个样例**
with torch.no_grad():
    model.set_processor(MODEL_NAME)
    # **修复 patch_size**
    model.processor.patch_size = 14
    print(f"[DEBUG] patch_size 确认: {model.processor.patch_size}")

    item = data_json[0]  # 只测试第一个样例
    query_text = item["question"]
    doc_name = item["doc_name"]
    doc_pages = dataset_df.loc[dataset_df['doc_name'] == doc_name]

    candidate_texts, candidate_images, passage_ids = [], [], []

    for _, row in doc_pages.iterrows():
        # **优先使用 vlm_text，其次 ocr_text**
        text = clean_text(row["vlm_text"]) if row["vlm_text"] else clean_text(row["ocr_text"]) if row[
            "ocr_text"] else None
        image_path = None  # 直接用空白图片替代

        candidate_texts.append(text)
        candidate_images.append(create_blank_image())
        passage_ids.append(row["passage_id"])

    print(f"[DEBUG] 处理后的候选文本数: {len(candidate_texts)}")
    print(f"[DEBUG] 处理后的候选图片数: {len(candidate_images)}")

    # **计算查询文本的 embedding**
    print("[DEBUG] 开始计算查询文本 embedding...")
    query_inputs = model.data_process(text=query_text, images=None, q_or_c="q",
                                      task_instruction="Retrieve the most relevant document page based on text matching.")
    query_embs = model(**query_inputs, output_hidden_states=True)[:, -1, :].to(device)
    query_embs = torch.nn.functional.normalize(query_embs, dim=-1)

    # **逐批处理候选页面**
    batch_size = 1  # 减小批处理大小
    candi_embs_list = []
    num_batches = len(candidate_texts) // batch_size + (1 if len(candidate_texts) % batch_size > 0 else 0)

    for i in range(num_batches):
        batch_texts = candidate_texts[i * batch_size: (i + 1) * batch_size]
        batch_images = [create_blank_image()] * len(batch_texts)  # 用空白图片填充

        print(f"[DEBUG] 处理批次 {i + 1}/{num_batches}...")
        try:
            candidate_inputs = model.data_process(text=batch_texts, images=batch_images, q_or_c="c")
            batch_candi_embs = model(**candidate_inputs, output_hidden_states=True)[:, -1, :].to(device)
            batch_candi_embs = torch.nn.functional.normalize(batch_candi_embs, dim=-1)
            candi_embs_list.append(batch_candi_embs)
        except Exception as e:
            print(f"[ERROR] 处理图片失败: {e}")
            torch.cuda.empty_cache()  # 释放显存
            continue

    # **合并所有 embedding**
    if candi_embs_list:
        candi_embs = torch.cat(candi_embs_list, dim=0)
    else:
        print("[ERROR] 无可用候选项")
        exit()

    # **计算相似度**
    scores = torch.matmul(query_embs, candi_embs.T)

    # **获取最佳匹配**
    best_match_idx = torch.argmax(scores).item()
    best_passage_id = passage_ids[best_match_idx]

    print(f"Query: {query_text}")
    print(f"Best Matched Passage ID: {best_passage_id}")
    print(f"Score: {scores[0, best_match_idx].item()}")
