import pandas as pd
import json


def merge_csv_files(file1, file2, output_file):
    """
    合并两个 CSV 文件，将 file2 追加到 file1 后面，并保存为新的 CSV 文件。
    并且将 passage_id 转换为 JSON 数组格式。

    :param file1: 第一个 CSV 文件的路径
    :param file2: 第二个 CSV 文件的路径
    :param output_file: 合并后的 CSV 文件的输出路径
    """
    # 读取 CSV 文件
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 确保 passage_id 是 JSON 数组格式
    df1["passage_id"] = df1["passage_id"].apply(lambda x: json.dumps(eval(x)))
    df2["passage_id"] = df2["passage_id"].apply(lambda x: json.dumps(eval(x)))

    # 合并数据
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # 保存合并后的数据
    merged_df.to_csv(output_file, index=False)
    print(f"合并完成，输出文件: {output_file}")


# 示例用法
if __name__ == "__main__":
    file1 = "m2kr/submission.csv"  # 第一个 CSV 文件路径
    file2 = "submission_mmdoc.csv"  # 第二个 CSV 文件路径
    output_file = "output/merged_output_flamingo_bilp2.csv"  # 输出文件路径

    merge_csv_files(file1, file2, output_file)
