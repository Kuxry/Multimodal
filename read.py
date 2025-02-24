from datasets import load_from_disk
import pandas as pd

# 加载已保存的数据集
ds_m2kr = load_from_disk("./m2kr")
ds_mmdoc = load_from_disk("./mmdoc")

# 取前 50 行并转换为 Pandas DataFrame
df_m2kr = pd.DataFrame(ds_m2kr["train"][:50])
df_mmdoc = pd.DataFrame(ds_mmdoc["train"][:50])

# 保存为 Excel 文件（去掉 encoding 参数）
df_m2kr.to_excel("M2KR_50rows.xlsx", index=False, engine="openpyxl")
df_mmdoc.to_excel("MMDocIR_50rows.xlsx", index=False, engine="openpyxl")

print("前50行数据已保存为 Excel 文件：M2KR_50rows.xlsx, MMDocIR_50rows.xlsx")
