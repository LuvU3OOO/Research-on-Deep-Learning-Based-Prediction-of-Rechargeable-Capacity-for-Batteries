import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

# 读取数据
path = r'../../dataHandle/Dataset.csv'
df = pd.read_csv(path)

# 删除不需要的列
# rm_cols = ["过充次数", "过放次数"]
# df = df.drop(labels=rm_cols, axis=1)

# 提取时间特征
df_time = pd.DataFrame({'date': pd.to_datetime(df['数据时间'])})

# # 1. 分离特征和标签
# X = df.iloc[:, 1:-1]  # 除去第一列时间和最后一列标签的其他特征
# y = df.iloc[:, -1]  # 最后一列是标签

X = df.iloc[:, 1:]  # 除去第一列时间特征
y = df.iloc[:, -1]  # 最后一列是标签
data = X

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# 用于存储处理后的数据
X_window = []
y_window = []

# 滑动窗口大小为5
window_size = 5

# 遍历数据
for i in range(len(X_scaled) - window_size):
    # 提取窗口内的特征
    window_features = X_scaled[i:i + window_size]
    # 提取预测目标
    target = y[i + window_size]

    # 将特征和目标分别添加到X和y中
    X_window.append(window_features)
    y_window.append(target)

# 转换为numpy数组
X_window = np.array(X_window)
y_window = np.array(y_window)
# x_test_0 = X_window[-200:-1]
# y_test_0 = y_window[-200:-1]
#
# # X_train = X_window[0:-100]
# # y_train = y_window[0:-100]
# # print(x_test_0[0],y_test_0[0])
# # 数据集分割为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_window, y_window, test_size=0.1)

# 手动按顺序分割数据集
test_size = 0.05
split_index = int(len(X_window) * (1 - test_size))
X_train, X_test = X_window[:-70], X_window[split_index:]
y_train, y_test = y_window[:-70], y_window[split_index:]
# 转换为Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# 创建自定义的 Dataset 类
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 创建训练集和测试集的 Dataset 对象
train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
test_dataset = CustomDataset(X_test_tensor, y_test_tensor)

# 使用 DataLoader 封装数据
batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(f'Training dataset sample shape: {train_dataset[1]}')
