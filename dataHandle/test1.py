import torch
import os
import pandas as pd
import numpy as np
from datetime import datetime

path = r"D:\Game\laji\广州琶洲算法大赛新能源汽车电池健康度预测数据集\15台车运行数据"

Cars = os.listdir(path)
print(Cars)


# 自定义函数，对特定的列进行不同的运算，返回一个 float 值
def custom_operation(col):
    if "状态" in col.name:
        res = col.mean()
    elif "电机转速" in col.name:
        res = (col.mean() / 10000)
    elif "电压" in col.name:
        res = (col.mean() / 1000)
    else:
        res = col.mean() / 100
    return res


for car in Cars:
    dir_path = os.path.join(path, car)
    files = os.listdir(dir_path)
    df = pd.DataFrame()
    for file_name in files:
        file_path = os.path.join(dir_path, file_name)
        print(file_name)
        sheets = pd.read_excel(file_path, sheet_name=None)
        # 计算每个DataFrame中每一列的平均值
        merged_df = pd.concat(
            [sheets['车辆运行数据'], sheets['驱动电机数据'].iloc[:, 1:], sheets['可充电储能装置数据'].iloc[:, 1:]],
            axis=1)
        # 对DataFrame进行线性插值
        merged_df = merged_df.interpolate(method='pad')
        df = pd.concat([df, merged_df], axis=0, ignore_index=True)

    df = df.sort_values(by='数据时间', ascending=True, ignore_index=True)
    # 需要删除的列
    rm_cols = ["可充电储能子系统各温度探针检测到的温度值", "单体电池电压", '数据时间']
    df = df.drop(labels=rm_cols, axis=1)

    # 删除包含None值的行
    df = df.dropna()
    print(df.shape)

    df_km = [df.iloc[0, 4].item()]
    df_soc = df.loc[(df["SOC"] >= 40) & (df["SOC"] <= 90) & (df["充电状态"] == 1)]
    df_soc = df_soc.reset_index(drop=True)
    df_soc.to_csv('test1.csv', index=False, encoding='utf-8')
    break
