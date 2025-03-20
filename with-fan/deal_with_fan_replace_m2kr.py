import pandas as pd

# 读取两个 CSV 文件
file_main = "updated_submission_2.csv"  # 主要文件
file_replace = "modified_merged_BGE_r_rank_top5.csv"  # 替换用的文件

# 加载数据
df_main = pd.read_csv(file_main)
df_replace = pd.read_csv(file_replace)

# 确保列名一致
df_main.columns = ["question_id", "passage_id"]
df_replace.columns = ["question_id", "passage_id"]

# 找出 `question_id >= 6414` 的部分
df_main_filtered = df_main[df_main["question_id"] >= 6414]# 6414 及以上的数据（保留）

# 找出 `question_id < 6414` 的部分（来自 `df_replace`）
df_replace_filtered = df_replace[df_replace["question_id"] < 6414] # 6414 之前的数据（替换）

# 合并两个部分
df_final = pd.concat([df_replace_filtered, df_main_filtered], ignore_index=True)

# 保存最终合并的 CSV 文件
output_file = "updated_submission_3.csv"
df_final.to_csv(output_file, index=False, quoting=2)  # quoting=2 确保 JSON 格式正确

print(f"✅ 处理完成，已保存到 {output_file}")
