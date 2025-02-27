import json
import pandas as pd

if __name__ == '__main__':
    # 1. 读取 `MMDocIR_doc_passages.parquet`
    dataset_df = pd.read_parquet('MMDocIR_doc_passages.parquet')

    # 2. 读取 `MMDocIR_gt_remove.jsonl`
    data_json = []
    with open("MMDocIR_gt_remove.jsonl", 'r', encoding="utf-8") as f:
        for line in f:
            data_json.append(json.loads(line.strip()))

    # 3. 关联查询（query）和文档页面数据
    for item in data_json:
        doc_name = item["doc_name"]  # 获取查询对应的文档名
        doc_pages = dataset_df.loc[dataset_df['doc_name'] == doc_name]  # 获取该文档的所有页面

        # 🟢 添加打印输出
        print(f"Query ID: {item['question_id']} - Query: {item['question']}")
        print(f"Matching Document: {doc_name}")
        print(f"Retrieved Passages: {doc_pages['passage_id'].tolist()[:5]}")  # 只显示前 5 个
        print("=" * 80)
