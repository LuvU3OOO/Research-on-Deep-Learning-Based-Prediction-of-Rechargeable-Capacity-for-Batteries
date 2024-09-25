import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# path = r"D:\Game\laji\广州琶洲算法大赛新能源汽车电池健康度预测数据集\15台车运行数据"
#
# Cars = os.listdir(path)
# print(Cars)
#
#
# # 定义一个函数，将字符串转换为数字列表，并计算平均值
# def calculate_average(string):
#     if type(string) == str:
#         # 将字符串分割成列表，并转换为整数或浮点数类型
#         numbers = list(map(float, string.split(',')))
#         # 计算列表中所有数字的平均值
#         average_value = round(sum(numbers) / len(numbers), 3)
#     else:
#         average_value = string
#     return average_value
#
#
# for car in Cars:
#     dir_path = os.path.join(path, car)
#     files = os.listdir(dir_path)
#     df_a = pd.DataFrame()
#     df_b = pd.DataFrame()
#     df_c = pd.DataFrame()
#     for file_name in files:
#         file_path = os.path.join(dir_path, file_name)
#         print(file_name)
#         sheets = pd.read_excel(file_path, sheet_name=None)
#         tmp_a = sheets['车辆运行数据']
#         tmp_b = sheets['驱动电机数据']
#         tmp_c = sheets['可充电储能装置数据']
#
#         df_a = pd.concat([df_a, tmp_a], axis=0, ignore_index=True)
#
#         df_b = pd.concat([df_b, tmp_b], axis=0, ignore_index=True)
#
#         df_c = pd.concat([df_c, tmp_c], axis=0, ignore_index=True)
#
#     # 对于列 A，对每个列表元素求平均值，并放回到原始 DataFrame 中
#     df_c['可充电储能子系统各温度探针检测到的温度值'] = df_c['可充电储能子系统各温度探针检测到的温度值'].apply(
#         calculate_average)
#     df_c['单体电池电压'] = df_c['单体电池电压'].apply(calculate_average)
#
#     df_a = df_a.sort_values(by='数据时间', ascending=True, ignore_index=True)
#     df_b = df_b.sort_values(by='数据时间', ascending=True, ignore_index=True)
#     df_c = df_c.sort_values(by='数据时间', ascending=True, ignore_index=True)
#
#     df_a = round(df_a, 2)
#     df_b = round(df_b, 2)
#     df_c = round(df_c, 2)
#     df_a.to_csv('../data/df_a_1.csv', index=False,encoding='utf-8')
#     # df_b.to_csv('../data/df_b.csv', index=False, encoding='utf-8')
#     # df_c.to_csv('../data/df_c.csv', index=False, encoding='utf-8')
#     break


path = r'Dataset.csv'
df = pd.read_csv(path, encoding='utf-8')

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 添加 DejaVu Sans 以支持负号
plt.rcParams['axes.unicode_minus'] = False  # 确保负号可以显示
#
# # 查看数据集的基本信息
# print(df.info())
#
#
# # 调整字体大小
# plt.rcParams['font.size'] = 14  # 全局字体大小
# plt.rcParams['axes.titlesize'] = 12  # 坐标轴标题字体大小
# plt.rcParams['axes.labelsize'] = 12  # 坐标轴标签字体大小
# plt.rcParams['xtick.labelsize'] = 10  # x轴刻度字体大小
# plt.rcParams['ytick.labelsize'] = 10  # y轴刻度字体大小
# plt.rcParams['legend.fontsize'] = 12  # 图例字体大小
#
# # 绘制热力图显示特征之间的相关性
# plt.figure(figsize=(15, 10))
# correlation_matrix = df.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Feature Correlation Heatmap')
# plt.savefig('heat_map.svg', format='svg')
# plt.show()


# 绘制每个特征的直方图
df.hist(bins=30, figsize=(16, 12), layout=(int(len(df.columns)/3)+1, 3))
plt.tight_layout()
# 保存图像为SVG格式
plt.savefig('histograms_d.svg', format='svg')
plt.show()
#
#
# path = r'Dataset.csv'
# df = pd.read_csv(path, encoding='utf-8')
#
# # 设置字体
# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 添加 DejaVu Sans 以支持负号
# plt.rcParams['axes.unicode_minus'] = False  # 确保负号可以显示
#
# # 查看数据集的基本信息
# print(df.info())
# # 计算描述性统计信息
# description = df.describe()
# description = round(description, 2)
# print(description)
# # 转置描述性统计信息
# description_transposed = description.transpose()
# # 保存到 csv 文件
# description_transposed.to_csv('description_transposed.csv')
