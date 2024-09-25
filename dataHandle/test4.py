# import pandas as pd
#
# # 示例数据
# soc_list = [pd.DataFrame({
#     '数据时间': ['2024-01-01', '2024-01-02', '2024-01-03'],
#     'A': [1, 2, 3],
#     'B': [4, 5, 6],
#     'C': [7, 8, 9]
# })]
#
# # 提取平均值并去除时间列
# tmp_df = soc_list[0].drop("数据时间", axis=1).mean().to_frame().T
#
# # 提取第一行时间值
# time_value = soc_list[0]['数据时间'].iloc[0]
#
# # 将时间值添加到新 DataFrame 的第一列
# tmp_df.insert(0, '数据时间', time_value)
#
# print(tmp_df)
import pandas as pd

# 示例数据
data = {'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8],
        'C': [9, 10, 11, 12]}
df = pd.DataFrame(data)

# 统计满足多个条件的行数
count = ((df['A'] < 3) & (df['B'] >= 6)).sum()

print("满足条件的行数:", count)
