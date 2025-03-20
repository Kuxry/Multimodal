import json
import pandas as pd
import csv  # 用于控制 CSV 输出格式

# 读取原始 CSV 文件
input_file_path = "merged_BGE_r_rank_top5.csv"
df = pd.read_csv(input_file_path, dtype=str, quotechar='"')  # 读取为字符串，防止数据类型错乱

# 确保 `question_id` 列存在（如果缺失则自动创建）
if "question_id" not in df.columns:
    df.insert(0, "question_id", range(len(df)))

# 确保 `passage_id` 列存在
if "passage_id" not in df.columns:
    raise ValueError("❌ 错误: CSV 文件缺少 'passage_id' 列！")


# 解析 `passage_id` 列，确保格式为 JSON 数组
def format_to_json_list(x):
    if isinstance(x, str) and x.strip():  # 确保非空
        try:
            # 先尝试 JSON 解析（标准格式）
            data = json.loads(x)
            if isinstance(data, list):
                return json.dumps([str(pid) for pid in data])  # 统一字符串格式
        except json.JSONDecodeError:
            pass  # 继续处理 Python 风格列表

        # 处理 Python 风格的列表（单引号）
        try:
            data = eval(x)  # 仅适用于受信数据
            if isinstance(data, list):
                return json.dumps([str(pid) for pid in data])  # 统一字符串格式
        except (SyntaxError, ValueError):
            return "[]"  # 解析失败，返回空列表

    return "[]"  # 处理空值


# 应用转换
df["passage_id"] = df["passage_id"].apply(format_to_json_list)

# 保存到新的 CSV 文件
output_file_path = "modified_merged_BGE_r_rank_top5.csv"
df.to_csv(output_file_path, index=False, quoting=csv.QUOTE_ALL, quotechar='"')  # 强制双引号

print(f"✅ 处理完成，已保存到 {output_file_path}")
