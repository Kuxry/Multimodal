import pandas as pd
from pathlib import Path

# 指定要合并的 CSV 文件路径
file_paths = [
    "modified_submission-401.csv",  # 替换为你的第一个文件路径
    "modified_submission-401-696.csv",  # 替换为你的第二个文件路径
    "modified_submission-697-10899.csv",    # 替换为你的第三个文件路径
    "modified_submission-10899-11657.csv"
]

# 读取并合并 CSV 文件
dfs = [pd.read_csv(file) for file in file_paths]

# 确保所有 DataFrame 具有相同的列名
for df in dfs:
    df.columns = ["question_id", "passage_id"]  # 确保列名一致

# 合并所有文件
df_merged = pd.concat(dfs, ignore_index=True)

# 保存合并后的文件
output_file = "merged_submission_all.csv"
df_merged.to_csv(output_file, index=False, quoting=2)  # quoting=2 确保 JSON 格式正确

print(f"✅ 合并完成，已保存到 {output_file}")
