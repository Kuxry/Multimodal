import torch
from transformers import AutoModel, AutoConfig,LlavaNextProcessor
from PIL import Image
import pandas as pd
import json
import re
import numpy as np
import os


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
IMAGE_BASE_PATH = "./"  # 图片存放的基础路径

dataset_df = pd.read_parquet(MMDocIR_doc_file)
data_json = []
for line in open(MMDocIR_gt_file, 'r', encoding="utf-8"):
    data_json.append(json.loads(line.strip()))

# 定义模型名称
MODEL_NAME = "BAAI/BGE-VL-MLLM-S2"

# 加载模型
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
model.to(device)


config = AutoConfig.from_pretrained(MODEL_NAME)
print(config)  # 输出完整的配置信息，查看是否包含 patch_size

def check_image_valid(image_path):
    """检查本地图片是否能成功打开，并确保尺寸非空"""
    full_path = os.path.join(IMAGE_BASE_PATH, image_path)
    print(f"[DEBUG] 尝试检查图片: {full_path}")

    if os.path.exists(full_path):
        try:
            img = Image.open(full_path)
            img.verify()  # 仅验证图片是否损坏，不加载到内存
            img = Image.open(full_path).convert("RGB")  # 重新打开
            width, height = img.size

            if width is None or height is None:
                print(f"[ERROR] 图片 {full_path} 尺寸异常，跳过")
                return None

            print(f"[DEBUG] 图片可用: {full_path}, 尺寸: {width}x{height}")
            return full_path  # 这里返回路径，而不是 `PIL.Image`
        except Exception as e:
            print(f"[ERROR] 读取图片失败: {full_path}, 错误: {e}")
            return None
    else:
        print(f"[ERROR] 图片不存在: {full_path}")
        return None  # 如果文件不存在，返回 None


with torch.no_grad():
    model.set_processor(MODEL_NAME)

    for item in data_json[:1]:  # 只测试第一个样例
        query_text = item["question"]
        doc_name = item["doc_name"]
        doc_pages = dataset_df.loc[dataset_df['doc_name'] == doc_name]

        # 修改后的候选数据收集
        candidate_texts = []
        candidate_images = []  # 存储图片路径
        passage_ids = []

        for _, row in doc_pages.iterrows():
            text = clean_text(row["vlm_text"]) if row["vlm_text"] else clean_text(row["ocr_text"]) if row[
                "ocr_text"] else "[EMPTY]"
            image_path = row["image_path"] if isinstance(row["image_path"], str) and pd.notna(
                row["image_path"]) else None

            candidate_texts.append(text)
            candidate_images.append(image_path)  # 存储路径
            passage_ids.append(row["passage_id"])

        # 过滤无效路径，并确保 `candidate_images` 里存储的是 **路径**
        valid_candidate_texts = []
        valid_candidate_images = []
        valid_passage_ids = []

        for i in range(len(candidate_images)):
            valid_path = check_image_valid(candidate_images[i]) if candidate_images[i] else None
            if valid_path:  # 确保路径有效
                valid_candidate_texts.append(candidate_texts[i])
                valid_candidate_images.append(valid_path)  # 存储路径，而不是 PIL.Image
                valid_passage_ids.append(passage_ids[i])

        print(f"[DEBUG] 处理后的候选图片数: {len(valid_candidate_images)}")

        # 处理查询输入
        query_inputs = model.data_process(
            text=query_text,
            images=None,  # 不传 images 避免错误
            q_or_c="q",
            task_instruction="Retrieve the most relevant document page based on text matching."
        )

        # 处理候选页面输入
        try:
            print(f"[DEBUG] 传递给 processor 的图片路径: {valid_candidate_images}")
            candidate_inputs = model.data_process(
                text=valid_candidate_texts,
                images=valid_candidate_images,  # 传递路径，而不是 PIL.Image
                q_or_c="c",
            )
        except Exception as e:
            print(f"[ERROR] 处理图片失败: {e}")
            raise

        # 计算嵌入向量
        query_embs = model(**query_inputs, output_hidden_states=True)[:, -1, :].to(device)
        candi_embs = model(**candidate_inputs, output_hidden_states=True)[:, -1, :].to(device)

        # 归一化嵌入向量
        query_embs = torch.nn.functional.normalize(query_embs, dim=-1)
        candi_embs = torch.nn.functional.normalize(candi_embs, dim=-1)

        # 计算相似度得分
        scores = torch.matmul(query_embs, candi_embs.T)

        # 获取最佳匹配 passage_id
        best_match_idx = torch.argmax(scores).item()
        best_passage_id = valid_passage_ids[best_match_idx]

        print(f"Query: {query_text}")
        print(f"Best Matched Passage ID: {best_passage_id}")
        print(f"Score: {scores[0, best_match_idx].item()}")
