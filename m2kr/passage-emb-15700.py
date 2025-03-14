import torch
import pandas as pd
import os
import re
from PIL import Image
from transformers import AutoModel
import warnings

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# ------------------------------
# 1. 设置设备、模型名称等
# ------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

MODEL_NAME = "BAAI/BGE-VL-MLLM-S2"
PASSAGE_FILE = "challenge_passage/train-00000-of-00001.parquet"  # 替换为你的文件路径
PASSAGE_IMAGE_DIR = "passage_images/Challenge"                   # 若没有图片，可忽略此路径

# 输出文件：保存前 15700 行的Passage向量
OUTPUT_EMB_FILE = "emb/passage_embeddings_first_15700.pt"

# ------------------------------
# 2. 加载模型
# ------------------------------
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
model.eval()
model.set_processor(MODEL_NAME)
model.processor.patch_size = 14

# ------------------------------
# 3. 载入 Passage 数据，并截取前15700行
# ------------------------------
df_passages = pd.read_parquet(PASSAGE_FILE)
print(f"[INFO] Total passages: {len(df_passages)}")
df_passages = df_passages.iloc[:15700]  # 仅保留前 15700 行
print(f"[INFO] Now using {len(df_passages)} passages for embedding")

# ------------------------------
# 4. 定义文本清洗函数
# ------------------------------
def clean_text(text):
    """清理文本，去除非 ASCII 字符以及多余的空格。"""
    if not isinstance(text, str):
        return "[EMPTY]"
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "[EMPTY]"

# ------------------------------
# 5. 定义函数：加载并切分图片上半部分
# ------------------------------
def load_and_split_top_half(image_path, base_dir):
    """
    1. 检查图片是否存在
    2. 验证图片是否能正常读取
    3. 切分并返回图片的上半部分（PIL Image）。
       如果任意环节失败，则返回 None。
    """
    if not isinstance(image_path, str):
        return None
    full_path = os.path.join(base_dir, image_path)
    if not os.path.exists(full_path):
        return None

    try:
        # 第一次打开只做 verify()
        with Image.open(full_path) as im_verify:
            im_verify.verify()

        # 第二次打开用于实际处理
        with Image.open(full_path) as img:
            w, h = img.size
            # 计算高度的一半
            half_h = h // 2
            # 这里示例仅返回“上半部分”图像
            top_half = img.crop((0, 0, w, half_h))
            return top_half

    except Exception as e:
        print(f"[ERROR] 加载/切分图片失败: {full_path}, 错误: {e}")
        return None

# ------------------------------
# 6. 对Passage文本 & 图片做预处理
# ------------------------------
df_passages["cleaned_content"] = df_passages["passage_content"].apply(clean_text)

# 原来是 check_image_valid，这里改成直接加载并切分出上半部分
df_passages["half_image"] = df_passages["page_screenshot"].apply(
    lambda x: load_and_split_top_half(x, PASSAGE_IMAGE_DIR)
)

passage_ids = df_passages["passage_id"].tolist()  # 用于后续查询时映射
all_texts = df_passages["cleaned_content"].tolist()
# 注意这里改成取 half_image
all_images = df_passages["half_image"].tolist()

# ------------------------------
# 7. 批量前向计算Passage向量
# ------------------------------
batch_size = 1  # 可根据显存或需求进行调整
num_passages = len(df_passages)
num_batches = (num_passages + batch_size - 1) // batch_size

print("[INFO] Start embedding passages in batches...")

emb_list = []
with torch.no_grad():
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_passages)

        batch_texts = all_texts[start_idx:end_idx]
        batch_imgs = all_images[start_idx:end_idx]
        batch_pids = passage_ids[start_idx:end_idx]  # 本批次对应的passage_id

        print(f"[INFO] Batch {i+1}/{num_batches}, passage_id range: {batch_pids[0]} ~ {batch_pids[-1]}")

        # data_process
        candidate_inputs = model.data_process(
            text=batch_texts,
            images=batch_imgs,  # 传入切好的上半部分图片
            q_or_c="c"          # 'c' for candidate passages
        )

        # 前向计算得到向量
        batch_embs = model(**candidate_inputs, output_hidden_states=True)[:, -1, :]
        # 归一化
        batch_embs = torch.nn.functional.normalize(batch_embs, dim=-1)

        # 搬回CPU以免占用显存
        batch_embs = batch_embs.cpu()
        emb_list.append(batch_embs)

        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            print(f"[INFO] Finished batch {i+1}/{num_batches}")

# 拼接所有批次的向量
passage_embs = torch.cat(emb_list, dim=0)  # shape: [num_passages, emb_dim]
del emb_list

print(f"[INFO] Completed. passage_embs shape = {passage_embs.shape}")

# ------------------------------
# 8. 保存向量 & ID 到文件
# ------------------------------
save_dict = {
    "passage_ids": passage_ids,
    "passage_embs": passage_embs,  # 在CPU上的tensor
}
torch.save(save_dict, OUTPUT_EMB_FILE)

print(f"[INFO] Passage embeddings saved to {OUTPUT_EMB_FILE}")
