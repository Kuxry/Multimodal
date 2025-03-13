import os
import re
import csv
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
from PIL import Image

# FSDP 主体
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# 如果要用混合精度, 需额外导入:
# from torch.distributed.fsdp import MixedPrecision
# from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

# OpenFlamingo
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms

###############################################################################
# (A) 函数: 推理逻辑 (相当于你原先的 retrieve_top_k_passages，但拆分成能按batch/多卡使用)
###############################################################################
def retrieve_top_k_passages(
    fsdp_model,
    tokenizer,
    image_processor,
    df_passages,
    question_id,
    instruction,
    question,
    img_path,
    image_dir="query_images/",
    device="cuda",
    top_k=5
):
    # 先把 instruction 当作 Query
    query_text = instruction

    # dummy 用于无图像时
    dummy_vision_x = torch.zeros(1, 1, 1, 3, 224, 224, device=device)

    # 若有图像，则生成更多文本
    if pd.notna(img_path):
        image_path = os.path.join(image_dir, img_path)
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")

            # 构造6D图像张量 (b, T_img, F, C, H, W)
            image_tensor = image_processor(image)  # (3, H, W)
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(1).unsqueeze(2).to(device)
            # => (1,1,1,3,H,W)

            # 做一次生成
            lang_x = tokenizer(["<image>" + instruction + "<|endofchunk|>"], return_tensors="pt", truncation=True, max_length=2048)
            input_ids = lang_x["input_ids"].to(device)
            attention_mask = lang_x["attention_mask"].to(device)

            with torch.no_grad():
                # ★ 注意: 使用 FSDP 包裹的模型，需要用 fsdp_model.module.generate(...)
                output = fsdp_model.module.generate(
                    vision_x=image_tensor,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=10
                )
            generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            query_text += " " + generated_text
        else:
            print(f"[WARNING] Image file not found: {image_path}")

    # 如有 question 字段，拼接
    if pd.notna(question):
        query_text += " " + question

    # 遍历所有 passages 进行评分
    scores = []
    for _, passage in df_passages.iterrows():
        passage_text = passage["passage_content"]
        input_text = f"Query: {query_text}\nPassage: {passage_text}\n相关性评分 (0-1):"

        lang_x = tokenizer(["<image>" + instruction + "<|endofchunk|>"], return_tensors="pt", truncation=True,
                           max_length=2048)
        input_ids = lang_x["input_ids"].to(device)
        attention_mask = lang_x["attention_mask"].to(device)

        with torch.no_grad():
            # 同理, 这里也要用 dummy vision_x
            output = fsdp_model.module.generate(
                vision_x=dummy_vision_x,
                lang_x=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10
            )
        generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        match = re.search(r"(\d+\.\d+)", generated_text)
        score = float(match.group(1)) if match else 0
        scores.append((score, passage["passage_id"]))

    # 排序并取 top_k
    sorted_passages = sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
    return [pid for _, pid in sorted_passages]

###############################################################################
# (B) 多进程入口: 每个 rank = [0..world_size-1] 执行
###############################################################################
def main_worker(rank, world_size):
    """
    rank: 当前进程编号
    world_size: 总进程数(与GPU数量对应)
    """
    # 1) 初始化进程组
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # 2) 绑定 GPU
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # 3) 创建模型 + 加载权重
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-7b",
        tokenizer_path="anas-awadalla/mpt-7b",
        cross_attn_every_n_layers=4,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # 下载checkpoint到本地
    checkpoint_path = hf_hub_download(
        repo_id="openflamingo/OpenFlamingo-9B-vitl-mpt7b",
        filename="checkpoint.pt"
    )
    # 加载权重到CPU再转GPU
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.lang_encoder.config.pad_token_id = tokenizer.eos_token_id

    # 4) 用 FSDP 包裹
    fsdp_model = FSDP(model, device_id=torch.cuda.current_device())

    if rank == 0:
        print(f"[Rank {rank}] FSDP model built & ready.")

    # 5) 读取数据 (每个进程都可读取同一个 parquet)
    df_passages = pd.read_parquet("challenge_passage/train-00000-of-00001.parquet")
    df_queries = pd.read_parquet("challenge_data/train-00000-of-00001.parquet")

    # 6) 将 queries 分给不同的 rank, 避免重复处理
    queries_list = df_queries.to_dict("records")
    total_q = len(queries_list)
    chunk_size = (total_q + world_size - 1) // world_size  # 向上取整
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, total_q)
    local_queries = queries_list[start_idx:end_idx]

    if rank == 0:
        print(f"[Rank 0] total {total_q} queries => each chunk_size ~ {chunk_size}")

    # 7) 推理 + 输出结果到 CSV
    #    为了简化，这里让每个rank输出自己的局部CSV, 后面可人工合并
    out_csv = f"submission_fsdp_rank{rank}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "passage_id"])

        for query in local_queries:
            question_id = query["question_id"]
            instruction = query["instruction"]
            question = query["question"] if pd.notna(query["question"]) else None
            img_path = query["img_path"] if pd.notna(query["img_path"]) else None

            top_passages = retrieve_top_k_passages(
                fsdp_model, tokenizer, image_processor, df_passages,
                question_id, instruction, question, img_path,
                image_dir="query_images/",
                device=device,
                top_k=5
            )
            # 写输出
            for pid in top_passages:
                writer.writerow([question_id, pid])

    if rank == 0:
        print(f"[Rank 0] Finished. Partial results => {out_csv}")

    # 8) 销毁进程组
    dist.destroy_process_group()


###############################################################################
# (C) 主函数: 用 mp.spawn 或 torchrun来启动多进程
###############################################################################
def run_fsdp_infer(world_size=4):
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    run_fsdp_infer(world_size=4)
