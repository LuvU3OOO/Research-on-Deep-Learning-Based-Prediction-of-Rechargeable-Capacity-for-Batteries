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
    if col.empty:
        return np.nan
    if "状态" in col.name:
        res = col.mean()
    elif "电机转速" in col.name:
        res = (col.mean() / 10000)
    elif "电压" in col.name:
        res = (col.mean() / 1000)
    else:
        res = col.mean() / 100
    return res


# 定义一个函数，将字符串转换为数字列表，并计算平均值
def calculate_average(string):
    if type(string) == str:
        # 将字符串分割成列表，并转换为整数或浮点数类型
        numbers = list(map(float, string.split(',')))
        # 计算列表中所有数字的平均值
        average_value = round(sum(numbers) / len(numbers), 3)
    else:
        average_value = string
    return average_value


for car in Cars:
    dir_path = os.path.join(path, car)
    files = os.listdir(dir_path)
    df = pd.DataFrame()
    df_dj = pd.DataFrame()
    for file_name in files:
        file_path = os.path.join(dir_path, file_name)
        print(file_name)
        sheets = pd.read_excel(file_path, sheet_name=None)
        # 计算每个DataFrame中每一列的平均值
        tmp_dj = sheets['驱动电机数据']
        merged_df = pd.concat(
            [sheets['车辆运行数据'], sheets['可充电储能装置数据'].iloc[:, 1:]],
            axis=1)
        df = pd.concat([df, merged_df], axis=0, ignore_index=True)
        df_dj = pd.concat([df_dj, tmp_dj], axis=0, ignore_index=True)
    df = df.sort_values(by='数据时间', ascending=True, ignore_index=True)
    df_dj = df_dj.sort_values(by='数据时间', ascending=True, ignore_index=True)
    # 对于列 A，对每个列表元素求平均值，并放回到原始 DataFrame 中
    df['可充电储能子系统各温度探针检测到的温度值'] = df['可充电储能子系统各温度探针检测到的温度值'].apply(
        calculate_average)
    df['单体电池电压'] = df['单体电池电压'].apply(calculate_average)

    # 需要删除的列
    # rm_cols = ["可充电储能子系统各温度探针检测到的温度值", "单体电池电压"]
    # df = df.drop(labels=rm_cols, axis=1)

    print(df.shape)

    df_km = [df.iloc[0, 4].item()]
    df_time = [df.iloc[0, 0]]
    # print(df_soc.head())
    soc_list = []  # 用于存储每个有效段落的列表
    start_index = None  # 记录当前段落起始索引
    df_soc = df.loc[(df["SOC"] >= 40) & (df["SOC"] <= 90) & (df["充电状态"] == 1)]
    df_soc = df_soc.reset_index(drop=True)
    # 遍历过滤后的DataFrame
    for i in range(len(df_soc)):
        soc_value = df_soc.iloc[i, 7]  # 当前SOC值
        I_charge = df_soc.iloc[i, 6]  # 当前电流
        prev_soc_value = df_soc.iloc[i - 1, 7] if i > 0 else None  # 前一个SOC值
        after_soc_value = df_soc.iloc[i + 1, 7] if i < len(df_soc) - 1 else None
        try:
            # 如果找到了以40开头的段落，并且前一个值不存在或大于40
            if soc_value <= 42 and (prev_soc_value > soc_value or prev_soc_value is None):
                start_index = i  # 设置段落起始索引

            # 如果找到了90结尾的值，并且当前段落已经开始
            elif soc_value >= 89 and start_index is not None and after_soc_value < soc_value:
                soc_list.append(df_soc.iloc[start_index:i + 1])  # 将段落加入列表
                df_km.append(df_soc.iloc[start_index, 4].item())
                df_time.append(df_soc.iloc[start_index, 0])
                start_index = None  # 重置起始索引，为下一个段落做准备
            elif after_soc_value is not None and after_soc_value < soc_value and start_index is not None:
                start_index = None
        except TypeError:
            print("出现错误：soc_value=", soc_value)
            continue  # 出现TypeError错误时跳过这一行
    print(len(soc_list))
    print(len(df_time))
    # print(soc_list[0])

    # 为了计算充电量，需要知道每次测量的时间间隔，dt=20秒
    dt = 20
    charge_quantity_list = []
    # 使用numpy的trapz函数计算安时积分
    for i in range(len(soc_list)):
        charge_quantity = np.trapz(abs(soc_list[i]['总电流']), dx=dt) / 3600
        charge_quantity_list.append(charge_quantity.round(3))
    # print(soc_list[i].iloc[0, 4], ":", charge_quantity)

    x_data_list = []
    for i in range(len(soc_list)):
        soc_list[i] = soc_list[i].drop(["SOC"], axis=1)
        tmp_df = soc_list[i].drop("数据时间", axis=1).mean().to_frame().T
        tmp_df = round(tmp_df, 3)
        time_value = soc_list[i]['数据时间'].iloc[0]
        # 将时间值添加到新 DataFrame 的第一列
        tmp_df.insert(0, '数据时间', time_value)
        x_data_list.append(tmp_df)

    x_data = pd.concat(x_data_list, ignore_index=True)

    print(x_data.shape)
    print(df_time[0],df_time[1])
    driver_list = []
    over_output_list = []
    over_charge_list = []
    for i in range(len(df_time) - 1):
        # 假设 df_km[i] 和 df_km[i+1] 是标量值，你可以使用 pd.Series() 将它们转换为 Series，并指定适当的索引
        # df_km_i = pd.Series(df_time[i], index=df["数据时间"].index)
        # df_km_i_plus_1 = pd.Series(df_time[i + 1], index=df["数据时间"].index)

        # 然后，你可以使用这些 Series 来执行比较操作
        df_drive = df.loc[(df["数据时间"] >= df_time[i]) & (df["数据时间"] < df_time[i+1])]
        dj_drive = df_dj.loc[(df["数据时间"] >= df_time[i]) & (df["数据时间"] < df_time[i+1])]
        # df_drive = df.loc[(df["充电状态"] != 1)&(df["累计里程"] >= df_km[i]) & (df["累计里程"] <= df_km[i+1])]
        df_drive = df_drive.drop(
            labels=["累计里程", "数据时间", "电池单体电压最高值", "电池单体电压最低值"],
            axis=1)
        dj_drive = dj_drive.drop(
            labels=["数据时间"],
            axis=1)
        # 统计满足多个条件的行数
        over_output_count = ((df_drive['SOC'] <= 20) & (df_drive['充电状态'] != 1)).sum()
        over_output_list.append(over_output_count)

        # # 计算每列的平均值，并将结果转置为一行
        # mean_values = df_drive.mean().transpose()
        # # 创建一个新的 DataFrame，将平均值作为一行数据
        # mean_df = pd.DataFrame(mean_values)
        # 应用自定义函数到 DataFrame 的每一列上，并得到一个 float 值
        result = df_drive.apply(custom_operation).sum()
        result_dj = dj_drive.apply(custom_operation).sum()
        print(result_dj)
        result = round(result+result_dj,3)
        driver_list.append(result)
    print(len(driver_list))

    # 将列表转换为 DataFrame，添加列名为 '驾驶习惯'
    dr_df = pd.DataFrame(driver_list, columns=['驾驶习惯'])
    charge_df = pd.DataFrame(charge_quantity_list, columns=['充电量'])
    over_charge_df = pd.DataFrame(over_charge_list, columns=['过充次数'])
    over_output_df = pd.DataFrame(over_output_list, columns=['过放次数'])
    x_data = pd.concat([x_data, dr_df], axis=1)
    data_csv_df = pd.concat([x_data, charge_df], axis=1)
    # 将 DataFrame 写入 CSV 文件
    data_csv_df.to_csv(f'../data/data_{car}.csv', index=False, encoding='utf-8')
    print(data_csv_df.shape)
