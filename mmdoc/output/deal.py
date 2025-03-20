import pandas as pd
import json

# 读取 CSV 文件
file_path = "mmdoc-10899-11657.csv"
df = pd.read_csv(file_path)

# 解析 passage_id：去掉方括号并转换为 Python 列表
df["passage_id"] = df["passage_id"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

# 确保所有元素转换为字符串，并转换回 JSON 格式字符串
df["passage_id"] = df["passage_id"].apply(lambda x: json.dumps([str(pid) for pid in x]))

# 保存到新的 CSV 文件
output_path = "modified_submission-10899-11657.csv.csv"
df.to_csv(output_path, index=False, quoting=2)  # quoting=2 确保字符串加引号

