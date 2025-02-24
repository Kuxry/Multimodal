import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset, concatenate_datasets
from PIL import Image

# -------------------------------
# 1. 加载数据集
# -------------------------------
# 加载 M2KR-Challenge 数据集（需要先登录 Hugging Face）
ds_m2kr = load_dataset("BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR", "CC_data")

# 加载 MMDocIR-Challenge 数据集
ds_mmdoc = load_dataset("MMDocIR/MMDocIR-Challenge")

# 假设两个数据集均有 'train' 和 'test' split，
# 我们将两者的训练集和测试集合并，以获得更丰富的数据
train_datasets = []
test_datasets = []

if "train" in ds_m2kr:
    train_datasets.append(ds_m2kr["train"])
if "train" in ds_mmdoc:
    train_datasets.append(ds_mmdoc["train"])

if "test" in ds_m2kr:
    test_datasets.append(ds_m2kr["test"])
if "test" in ds_mmdoc:
    test_datasets.append(ds_mmdoc["test"])

if len(train_datasets) > 1:
    train_dataset = concatenate_datasets(train_datasets)
else:
    train_dataset = train_datasets[0]

if len(test_datasets) > 1:
    test_dataset = concatenate_datasets(test_datasets)
else:
    test_dataset = test_datasets[0]

# -------------------------------
# 2. 初始化 CLIP 模型和处理器
# -------------------------------
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# -------------------------------
# 3. 定义 collate_fn 处理函数
# -------------------------------
def collate_fn(batch):
    texts = []
    images = []
    for item in batch:
        # 根据数据集实际字段名称调整，以下假设每个样本包含 "text" 和 "image" 字段
        texts.append(item.get("text", ""))
        img_item = item.get("image", None)
        # 如果图像为文件路径，则加载图像
        if isinstance(img_item, str):
            if os.path.exists(img_item):
                image = Image.open(img_item).convert("RGB")
            else:
                raise FileNotFoundError(f"Image file not found: {img_item}")
        else:
            image = img_item  # 假设已为 PIL.Image 对象
        images.append(image)

    # 利用 CLIPProcessor 同时预处理文本和图像
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    return inputs


# -------------------------------
# 4. 构造 DataLoader
# -------------------------------
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# -------------------------------
# 5. 定义优化器和损失函数
# -------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# -------------------------------
# 6. 训练过程
# -------------------------------
num_epochs = 3  # 根据实际情况调整 Epoch 数量

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        # 将 batch 中所有 tensor 转移到指定设备
        for key in batch:
            batch[key] = batch[key].to(device)

        # 前向传播得到输出，CLIP 输出包含 logits_per_text 与 logits_per_image
        outputs = model(**batch)
        logits = outputs.logits_per_text  # shape: [batch_size, batch_size]

        # 对于每个样本，其正例为对角线上的索引
        labels = torch.arange(logits.size(0)).to(device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

# -------------------------------
# 7. 测试/评估过程
# -------------------------------
model.eval()
all_logits = []
with torch.no_grad():
    for batch in test_loader:
        for key in batch:
            batch[key] = batch[key].to(device)

        outputs = model(**batch)
        logits = outputs.logits_per_text  # shape: [batch_size, batch_size]
        all_logits.append(logits.cpu())

all_logits = torch.cat(all_logits, dim=0)
print("Test logits shape:", all_logits.shape)
