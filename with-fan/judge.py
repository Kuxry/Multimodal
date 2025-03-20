import pandas as pd

# 读取原始 CSV 文件
file_path = "merged_submission_all.csv"
output_file = "merged_submission_all_deduplicated.csv"

# 读取数据
df = pd.read_csv(file_path, dtype=str)

# 确保 `question_id` 列存在
if "question_id" not in df.columns:
    raise ValueError("❌ 错误: CSV 文件中缺少 'question_id' 列！")

# 🚀 去重：按 `question_id` 只保留最后出现的那一行
df = df.drop_duplicates(subset=["question_id"], keep="last")

# 保存去重后的文件
df.to_csv(output_file, index=False, quoting=1)  # quoting=1 以防止格式错误

print(f"✅ 去重完成，已保存到 {output_file}")
