import json
import ast
import pandas as pd
# 读取原始 CSV 文件
input_file_path = "../emb/retrieval_results.csv"
df = pd.read_csv(input_file_path)

# 将 "query_id" 列重命名为 "question_id"（如果存在）
if "query_id" in df.columns:
    df.rename(columns={"query_id": "question_id"}, inplace=True)

# 确保 "question_id" 存在，否则创建一个从 0 开始的索引
if "question_id" not in df.columns:
    df.insert(0, "question_id", range(len(df)))

# 解析 "top_passage_ids" 列，确保格式匹配 JSON
def safe_json_loads(x):
    if isinstance(x, str) and x.strip():
        try:
            # 处理 Python 风格的列表（单引号）
            if x.startswith("[") and x.endswith("]") and "'" in x:
                return ast.literal_eval(x)  # 转换为 Python 列表
            return json.loads(x)  # 直接解析 JSON
        except (json.JSONDecodeError, ValueError, SyntaxError):
            return []  # 解析失败返回空列表
    return []  # 处理 NaN 或空值

# 应用转换
df["top_passage_ids"] = df["top_passage_ids"].apply(safe_json_loads)

# 确保所有元素转换为字符串，并转换回 JSON 格式字符串
df["top_passage_ids"] = df["top_passage_ids"].apply(lambda x: json.dumps([str(pid) for pid in x]))

# 重命名列以匹配目标格式
df.rename(columns={"top_passage_ids": "passage_id"}, inplace=True)

# 保存到新的 CSV 文件
output_file_path = "modified_submission-m2kr.csv"
df.to_csv(output_file_path, index=False, quoting=2)  # quoting=2 确保 JSON 字符串正确
