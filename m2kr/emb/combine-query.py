import torch

# 1. 依次加载两个 .pt 文件
data_part1 = torch.load("query_embeddings_challenge_data_part1.pt")
data_part2 = torch.load("query_embeddings_challenge_data_part2.pt")

# 2. 取出ID列表与向量
ids_part1 = data_part1["query_ids"]     # list[str]
embs_part1 = data_part1["query_embs"]   # shape [N1, D]

ids_part2 = data_part2["query_ids"]     # list[str]
embs_part2 = data_part2["query_embs"]   # shape [N2, D]

# 3. 拼接
merged_ids = ids_part1 + ids_part2
merged_embs = torch.cat([embs_part1, embs_part2], dim=0)  # shape [N1+N2, D]

print(f"[INFO] merged_ids: {len(merged_ids)}")
print(f"[INFO] merged_embs: {merged_embs.shape}")

# 4. 保存为新的 .pt 文件
save_dict = {
    "query_ids": merged_ids,
    "query_embs": merged_embs
}
torch.save(save_dict, "query_embeddings_challenge_data_merged.pt")

print("[INFO] Done. Merged file => query_embeddings_challenge_data_merged.pt")
