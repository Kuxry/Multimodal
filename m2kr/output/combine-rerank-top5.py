import pandas as pd
import os

# 设定文件路径
file1 = "../submission_bge_r_rank_top5_2420.csv"
file2 = "../submission_bge_r_rank_top5_2420-4420.csv"
file3 = "../submission_bge_r_rank_top5_4420-6420.csv"

# 读取 CSV 文件
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# 合并 CSV 文件（按行合并）
df_merged = pd.concat([df1, df2, df3], ignore_index=True)

# 输出合并后的 CSV 文件
output_file = "merged_r_rank_top5.csv"
df_merged.to_csv(output_file, index=False)

print(f"合并完成，文件已保存为 {output_file}")
