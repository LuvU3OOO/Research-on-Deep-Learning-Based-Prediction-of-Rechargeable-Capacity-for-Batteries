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
hidden_dim = 256
n_epochs = 200
num_layers = 1
learning_rate = 1e-3
weight_decay = 1e-6
is_bidirectional = True
dropout_prob = 0
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
embed_dim = 256

if is_bidirectional:
    D = 2
else:
    D = 1


class LstmModel(nn.Module):
    def __init__(self):
        super(LstmModel, self).__init__()

        # dimension for LSTM or BiLSTM

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=is_bidirectional,
                            dropout=dropout_prob,
                            batch_first=True
                            )

        self.fc = nn.Linear(hidden_dim * D, output_dim)

    def forward(self, x):
        hidden_0 = torch.zeros(D * num_layers, x.size(0), hidden_dim).to(device)
        c_0 = torch.zeros(D * num_layers, x.size(0), hidden_dim).to(device)

        output, (h_n, c_n) = self.lstm(x, (hidden_0.detach(), c_0.detach()))

        output = self.fc(output[:, -1, :])

        return output


# Initialize model, loss function, and optimizer
LstmModel = LstmModel().to(device)
optimizer = torch.optim.Adam(LstmModel.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = torch.nn.MSELoss(reduction="mean")

# Train model
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = LstmModel(inputs)
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
df = pd.DataFrame(train_losses, columns=["BiLSTM"])
# 保存到 CSV 文件
df.to_csv('loss_BiLSTM.csv', index=False)
# Model prediction and evaluation
LstmModel.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = LstmModel(inputs)
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
df = pd.DataFrame(y_pred, columns=["BILSTM"])

# 保存到 CSV 文件
df.to_csv('pred_BILSTM.csv', index=False)
