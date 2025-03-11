import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModel, AutoConfig, LlavaNextProcessor
from PIL import Image
import pandas as pd
import json
import re
import numpy as np
import os

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
IMAGE_BASE_PATH = "./"  # 图片存放的基础路径

dataset_df = pd.read_parquet(MMDocIR_doc_file)
data_json = [json.loads(line.strip()) for line in open(MMDocIR_gt_file, 'r', encoding="utf-8")]

# 定义模型名称
MODEL_NAME = "BAAI/BGE-VL-MLLM-S2"


def setup(rank, world_size):
    """ 初始化分布式训练 """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """ 清理分布式环境 """
    dist.destroy_process_group()


def process_batch(rank, world_size, candidate_texts, candidate_images, passage_ids, query_embs):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    model.eval()
    model.set_processor(MODEL_NAME)
    model.processor.patch_size = 14

    batch_size = 2  # 适当增大批处理，提高效率
    local_indices = list(range(rank, len(candidate_texts), world_size))
    local_texts = [candidate_texts[i] for i in local_indices]
    local_images = [candidate_images[i] for i in local_indices]
    local_passage_ids = [passage_ids[i] for i in local_indices]

    local_candi_embs_list = []
    for batch_start in range(0, len(local_texts), batch_size):
        batch_texts = local_texts[batch_start: batch_start + batch_size]
        batch_images = local_images[batch_start: batch_start + batch_size]

        print(f"[GPU {rank}] 处理 batch {batch_start // batch_size + 1}/{len(local_texts) // batch_size}")

        try:
            torch.cuda.empty_cache()
            candidate_inputs = model.data_process(text=batch_texts, images=batch_images, q_or_c="c")
            batch_candi_embs = model(**candidate_inputs, output_hidden_states=True)[:, -1, :].to(device)
            batch_candi_embs = torch.nn.functional.normalize(batch_candi_embs, dim=-1)
            local_candi_embs_list.append(batch_candi_embs)
        except Exception as e:
            print(f"[GPU {rank}] [ERROR] 处理失败: {e}")
            torch.cuda.empty_cache()
            continue

    if local_candi_embs_list:
        local_candi_embs = torch.cat(local_candi_embs_list, dim=0)
    else:
        print(f"[GPU {rank}] 无可用候选项")
        cleanup()
        return None, None

    candi_embs_list = [torch.zeros_like(local_candi_embs) for _ in range(world_size)]
    dist.all_gather(candi_embs_list, local_candi_embs)
    final_candi_embs = torch.cat(candi_embs_list, dim=0)

    scores = torch.matmul(query_embs, final_candi_embs.T)
    top_k = 5
    top_indices = torch.topk(scores, top_k, dim=1).indices.squeeze(0).tolist()
    top_passage_ids = [passage_ids[idx] for idx in top_indices]
    top_scores = [scores[0, idx].item() for idx in top_indices]

    if rank == 0:
        print("Top 5 Best Matched Passage IDs and Scores:")
        for pid, score in zip(top_passage_ids, top_scores):
            print(f"Passage ID: {pid}, Score: {score}")

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    query_text = data_json[0]["question"]
    doc_name = data_json[0]["doc_name"]
    doc_pages = dataset_df.loc[dataset_df['doc_name'] == doc_name]

    candidate_texts, candidate_images, passage_ids = [], [], []
    for _, row in doc_pages.iterrows():
        text = clean_text(row["vlm_text"]) if row["vlm_text"] else clean_text(row["ocr_text"]) if row[
            "ocr_text"] else None
        image_path = row["image_path"] if isinstance(row["image_path"], str) and pd.notna(row["image_path"]) else None
        candidate_texts.append(text)
        candidate_images.append(image_path)
        passage_ids.append(row["passage_id"])

    with torch.no_grad():
        query_inputs = model.data_process(text=query_text, images=None, q_or_c="q",
                                          task_instruction="Retrieve the most relevant document page based on text matching.")
        query_embs = model(**query_inputs, output_hidden_states=True)[:, -1, :].to("cuda:0")
        query_embs = torch.nn.functional.normalize(query_embs, dim=-1)

    mp.spawn(process_batch, args=(world_size, candidate_texts, candidate_images, passage_ids, query_embs),
             nprocs=world_size, join=True)
