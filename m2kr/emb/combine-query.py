import torch
import pandas as pd
import csv

# ====================================================
# 1. 加载 & 合并 Query Embeddings (若分2个文件，可拼接)
# ====================================================
# 假设你有两个文件: query_embeddings_challenge_data_part1.pt 与 part2.pt
# 如果只有一个文件，直接 load 就行。
data_q1 = torch.load("query_embeddings_challenge_data_part1.pt")
data_q2 = torch.load("query_embeddings_challenge_data_part2.pt")

query_ids_1 = data_q1["query_ids"]       # list[str]
query_embs_1 = data_q1["query_embs"]     # Tensor, shape [N1, D]

query_ids_2 = data_q2["query_ids"]       # list[str]
query_embs_2 = data_q2["query_embs"]     # Tensor, shape [N2, D]

# 合并
all_query_ids = query_ids_1 + query_ids_2
all_query_embs = torch.cat([query_embs_1, query_embs_2], dim=0)  # [N1+N2, D]

print(f"[INFO] Merged Query Embeddings: {all_query_embs.shape}, total {len(all_query_ids)} queries.")

# ====================================================
# 2. 加载 & 合并 Passage Embeddings (若也分了多文件)
# ====================================================
# 如果你把 Passages 也分了多个 .pt 文件，那么同理可做拼接；
# 假设这里我们只示范加载一个合并后的 'passage_merged.pt'。
data_passage = torch.load("passage_merged.pt")
all_passage_ids = data_passage["passage_ids"]      # list[str] or list[int]
all_passage_embs = data_passage["passage_embs"]    # Tensor, shape [P, D]

print(f"[INFO] Passage Embeddings: {all_passage_embs.shape}, total {len(all_passage_ids)} passages.")

# ====================================================
# 3. 将向量放到 GPU，计算相似度
# ====================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

query_embs_gpu = all_query_embs.to(device)       # [Q, D]
passage_embs_gpu = all_passage_embs.to(device)   # [P, D]

scores = torch.matmul(query_embs_gpu, passage_embs_gpu.T)  # [Q, P]
print("[INFO] Similarity matrix shape:", scores.shape)

# ====================================================
# 4. 取每条 Query 的 TopK 并输出
# ====================================================
top_k = 5
top_scores, top_indices = torch.topk(scores, top_k, dim=1)  # [Q, top_k]

# 你可以将检索结果保存到 CSV
output_csv = "retrieval_results.csv"
with open(output_csv, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["query_id", "top_passage_ids"])

    for i, qid in enumerate(all_query_ids):
        # 取出该 Query 的前K个索引
        best_indices = top_indices[i].tolist()  # [top_k]
        best_pids = [all_passage_ids[idx] for idx in best_indices]

        # 写入 CSV
        writer.writerow([qid, str(best_pids)])

print(f"[INFO] Retrieval results saved to {output_csv}")
