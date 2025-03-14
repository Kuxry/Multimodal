import torch
from transformers import AutoModel
from PIL import Image
import pandas as pd
import json
import re
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
data_json = [json.loads(line.strip()) for line in open(MMDocIR_gt_file, 'r', encoding="utf-8")]

# 定义模型名称
MODEL_NAME = "BAAI/BGE-VL-MLLM-S2"

# 加载模型
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
model.eval()


# **检查图片有效性**
def check_image_valid(image_path):
    """检查图片是否可用，并返回有效路径"""
    full_path = os.path.join(IMAGE_BASE_PATH, image_path)
    if os.path.exists(full_path):
        try:
            img = Image.open(full_path)
            img.verify()  # 仅验证
            img = Image.open(full_path).convert("RGB")  # 重新打开
            return full_path
        except Exception as e:
            print(f"[ERROR] 读取图片失败: {full_path}, 错误: {e}")
    return None


#697-1657
data_json = data_json[697:1657]

results = []
output_file = "mmdoc-697-1657.csv"

# 创建 CSV 并写入标题
pd.DataFrame(columns=["question_id", "passage_id"]).to_csv(output_file, index=False, encoding="utf-8")

with torch.no_grad():
    model.set_processor(MODEL_NAME)
    model.processor.patch_size = 14

    for idx, item in enumerate(data_json):
        query_text = item["question"]
        question_id = item["question_id"]
        print(f"[INFO] 处理 question_id: {question_id} ({idx + 697}/1657)")
        doc_name = item["doc_name"]
        doc_pages = dataset_df.loc[dataset_df['doc_name'] == doc_name]

        candidate_texts, candidate_images, passage_ids = [], [], []

        for _, row in doc_pages.iterrows():
            text = clean_text(row["vlm_text"]) if row["vlm_text"] else clean_text(row["ocr_text"]) if row[
                "ocr_text"] else None
            image_path = row["image_path"] if isinstance(row["image_path"], str) and pd.notna(
                row["image_path"]) else None
            valid_image_path = check_image_valid(image_path) if image_path else None

            candidate_texts.append(text)
            candidate_images.append(valid_image_path)
            passage_ids.append(row["passage_id"])

        # **计算查询文本的 embedding**
        query_inputs = model.data_process(text=query_text, images=None, q_or_c="q",
                                          task_instruction="Retrieve the most relevant document page based on text matching.")
        query_embs = model(**query_inputs, output_hidden_states=True)[:, -1, :].to(device)
        query_embs = torch.nn.functional.normalize(query_embs, dim=-1)

        # **逐批处理候选页面**
        batch_size = 1
        candi_embs_list = []
        num_batches = len(candidate_texts) // batch_size + (1 if len(candidate_texts) % batch_size > 0 else 0)

        for i in range(num_batches):
            batch_texts = candidate_texts[i * batch_size: (i + 1) * batch_size]
            batch_images = candidate_images[i * batch_size: (i + 1) * batch_size]

            try:
                torch.cuda.empty_cache()
                candidate_inputs = model.data_process(text=batch_texts, images=batch_images, q_or_c="c")
                batch_candi_embs = model(**candidate_inputs, output_hidden_states=True)[:, -1, :].to(device)
                batch_candi_embs = torch.nn.functional.normalize(batch_candi_embs, dim=-1)
                candi_embs_list.append(batch_candi_embs)
            except Exception as e:
                print(f"[ERROR] 处理图片失败: {e}")
                torch.cuda.empty_cache()
                continue

        if candi_embs_list:
            candi_embs = torch.cat(candi_embs_list, dim=0)
        else:
            print(f"[ERROR] 无可用候选项 for question_id: {question_id}")
            continue

        # **计算相似度**
        scores = torch.matmul(query_embs, candi_embs.T)

        # **获取前5个最佳匹配 passage_id**
        top_k = 5
        top_indices = torch.topk(scores, top_k, dim=1).indices.squeeze(0).tolist()
        top_passage_ids = [passage_ids[idx] for idx in top_indices]

        # **保证 passage_id 数量不超过 5，并转换为字符串格式**
        top_passage_ids_str = "[" + ",".join(map(str, top_passage_ids)) + "]"

        # **保存结果到 CSV 文件**
        pd.DataFrame([[question_id, top_passage_ids_str]], columns=["question_id", "passage_id"]).to_csv(
            output_file, mode='a', header=False, index=False, encoding="utf-8"
        )