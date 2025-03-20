import torch
import csv

# ========== 1. 加载 Query Embeddings ==========
# 假设你已将前后两部分合并到一个文件 "query_embeddings_challenge_data_merged.pt"
query_data = torch.load("query_embeddings_challenge_data_merged.pt")
query_ids = query_data["query_ids"]         # e.g. list[str]
query_embs = query_data["query_embs"]       # shape: [Q, D]
print(f"[INFO] Loaded Query Embeddings => shape={query_embs.shape}, count={len(query_ids)}")

# ========== 2. 加载 Passage Embeddings ==========
# 假设你也有一个合并好的文件 "passage_embeddings_merged.pt"
passage_data = torch.load("passage_merged.pt")
passage_ids = passage_data["passage_ids"]    # list[str] or list[int]
passage_embs = passage_data["passage_embs"]  # shape: [P, D]
print(f"[INFO] Loaded Passage Embeddings => shape={passage_embs.shape}, count={len(passage_ids)}")

# ========== 3. 设置设备 & 放到GPU（可选）==========
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

query_embs_gpu = query_embs.to(device)      # [Q, D]
passage_embs_gpu = passage_embs.to(device)  # [P, D]

# ========== 4. 相似度计算 & 取TopK ==========
print("[INFO] Performing similarity calculation via matrix multiplication...")
# (Q, D) x (D, P) => (Q, P)
scores = torch.matmul(query_embs_gpu, passage_embs_gpu.T)

top_k = 100
print(f"[INFO] Retrieving top {top_k} passages for each query...")
top_scores, top_indices = torch.topk(scores, top_k, dim=1)  # shape: [Q, K] (scores & indices)

# ========== 5. 保存结果到CSV（或打印） ==========
output_csv = "retrieval_results-top100.csv"
with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["query_id", "top_passage_ids"])

    for i, qid in enumerate(query_ids):
        # 取该 Query 的 top_k Passage 索引
        best_indices = top_indices[i].tolist()  # 长度=top_k
        # 映射回对应的 passage_id
        best_pids = [passage_ids[idx] for idx in best_indices]

        # 写入CSV，每行示例: query_id, [passage_id1, passage_id2, ...]
        writer.writerow([qid, str(best_pids)])

print(f"[INFO] Retrieval results saved to: {output_csv}")
