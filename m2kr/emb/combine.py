import torch

# 1. 依次加载三个文件
data1 = torch.load("passage_embeddings_1_15700.pt")  # 例如包含 "passage_ids" 和 "passage_embs"
data2 = torch.load("passage_embeddings_2_15701-31000.pt")
data3 = torch.load("passage_embeddings_3_31001-47318.pt")

# 2. 分别取出 ID 列表和 Embedding 张量
ids1, embs1 = data1["passage_ids"], data1["passage_embs"]
ids2, embs2 = data2["passage_ids"], data2["passage_embs"]
ids3, embs3 = data3["passage_ids"], data3["passage_embs"]

# 3. 在 CPU 上进行拼接（如果当前已经在 GPU，可以先把 embsX = embsX.cpu()）
# 先把 passage_embs 都放到一个列表里，再用 torch.cat() 连接
merged_ids = ids1 + ids2 + ids3
merged_embs = torch.cat([embs1, embs2, embs3], dim=0)  # [N1 + N2 + N3, emb_dim]

# 4. 存到一个新的文件，方便后续统一加载
torch.save({
    "passage_ids": merged_ids,
    "passage_embs": merged_embs
}, "passage_merged.pt")

print("合并完成，已保存到 passage_merged.pt")
print("合并后 embedding shape =", merged_embs.shape)
print("合并后 passage_ids 数量 =", len(merged_ids))
