import os
from sklearn.metrics import mean_squared_error
import pandas as pd
from Train.utils.window_slip import train_loader, test_loader
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# parameters
input_dim = 13
n_seq = 5
batch_size = 32
output_dim = 1
hidden_dim = 512
n_epochs = 200
num_layers = 1
learning_rate = 1e-4
weight_decay = 1e-6
is_bidirectional = True
dropout_prob = 0
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
embed_dim = 512

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.linear = nn.Linear(input_dim, embed_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


class Transformer(nn.Module):
    def __init__(self):  # d_model 表示特征维度（必须是head的整数倍）, num_layers 表示 Encoder_layer 的层数， dropout 用于防止过你和
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        # self.fc = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(input_dim,hidden_dim)  # 位置编码前要做归一化，否则捕获不到位置信息
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout_prob)  # 这里用了八个头
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # self.fc1 = nn.Linear(hidden_dim, 32)
        self.decoder = nn.Linear(hidden_dim, 1)  # 这里用全连接层代替了decoder， 其实也可以加一下Transformer的decoder试一下效果
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        # src = self.fc(src)
        src = self.pos_encoder(src)
        mask = self._generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, mask)
        # output = self.fc1(output)
        output = self.decoder(output[:, -1, :])
        return output



# 初始化模型、损失函数和优化器
TfModel = Transformer().to(device)
optimizer = torch.optim.Adam(TfModel.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss(reduction="mean")

# 训练模型
train_losses = []
val_losses = []

for epoch in range(n_epochs):

    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = TfModel(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    print(f'Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_losses[-1]:.4f}')


# Plot loss
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_lstm.svg", format='svg')
plt.show()
# 将 numpy 数组转换为 DataFrame
df = pd.DataFrame(train_losses, columns=["Transformer"])
# 保存到 CSV 文件
df.to_csv('loss_Transformer.csv', index=False)
# 模型预测和评估
TfModel.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = TfModel(inputs)
        y_pred.extend(outputs.squeeze().cpu().numpy())
        y_true.extend(targets.cpu().numpy())


# Compute evaluation metrics
y_pred = np.array(y_pred)
y_true = np.array(y_true)
# 将 numpy 数组转换为 DataFrame
# df = pd.DataFrame(y_true, columns=["True"])
#
# # 保存到 CSV 文件
# df.to_csv('true.csv', index=False)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
r2 = r2_score(y_true, y_pred)

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'R2: {r2:.2f}')
# 绘制预测值和真实值的折线图进行对比
plt.figure(figsize=(15, 5))
plt.plot(y_true, label='True Values', color='blue')
plt.plot(y_pred, label='Predictions', color='red', linestyle='dashed')
plt.title('Comparison of True Values and Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.legend()
plt.show()

# 将 numpy 数组转换为 DataFrame
df = pd.DataFrame(y_pred, columns=["Transformer"])

# 保存到 CSV 文件
df.to_csv('pred_Transformer.csv', index=False)
