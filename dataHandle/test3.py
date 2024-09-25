import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取数据
path = r'../data'
dfs = []
files = os.listdir(path)

for file in files:
    file_path = os.path.join(path, file)
    df = pd.read_csv(file_path)
    dfs.append(df)

dataset_df = pd.concat(dfs, ignore_index=True)

# 删除不需要的列
dataset_df = dataset_df.drop(labels=["充电状态", "车速","车辆状态"], axis=1)
dataset_df = dataset_df.sort_values(by='数据时间', ascending=True, ignore_index=True)
dataset_df.to_csv("Dataset.csv", index=False)


