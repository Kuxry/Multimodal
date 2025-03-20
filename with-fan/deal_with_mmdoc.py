import pandas as pd

# 读取 CSV 文件
file1_path = "search_results-fang.csv"  # 第一个文件路径（主文件）
file2_path = "merged_submission_all.csv"  # 第二个文件路径（用于替换）
output_file = "updated_submission_2.csv"  # 结果文件路径

# 读取数据
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# 确保列名一致
df1.columns = ["question_id", "passage_id"]
df2.columns = ["question_id", "passage_id"]

# 使用 `question_id` 作为索引进行替换
df1.set_index("question_id", inplace=True)  # 设置 question_id 为索引
df2.set_index("question_id", inplace=True)

# 更新 df1 中的 `passage_id`，如果 df2 中存在相应的 `question_id`
df1.update(df2)

# 重置索引并保存结果
df1.reset_index(inplace=True)
df1.to_csv(output_file, index=False, quoting=2)  # quoting=2 确保 JSON 格式正确

print(f"✅ 替换完成，已保存到 {output_file}")
