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
is_bidirectional = False
dropout_prob = 0
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
embed_dim = 256

# Define GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_prob,
            batch_first=True
        )
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        hidden_0 = torch.zeros(num_layers, src.size(0), hidden_dim).to(device)
        x, h_n = self.gru(src, hidden_0.detach())
        output = self.decoder(x[:, -1, :])
        return output

# Initialize model, loss function, and optimizer
GruModel = GRUModel(input_dim, hidden_dim, num_layers, output_dim, dropout_prob).to(device)
optimizer = torch.optim.Adam(GruModel.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = torch.nn.MSELoss(reduction="mean")

# Train model
train_losses = []

for epoch in range(n_epochs):
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = GruModel(inputs)
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
plt.savefig("loss_gru.svg", format='svg')
plt.show()

# Model prediction and evaluation
GruModel.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = GruModel(inputs)
        y_pred.extend(outputs.squeeze().cpu().numpy())
        y_true.extend(targets.cpu().numpy())

# Compute evaluation metrics
y_pred = np.array(y_pred)
y_true = np.array(y_true)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
r2 = r2_score(y_true, y_pred)

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'R2: {r2:.2f}')

# Plot comparison of true values and predictions
plt.figure(figsize=(18, 8))
plt.plot(y_true, label='True Values', color='blue', linestyle='-', marker='o', markersize=4)
plt.plot(y_pred, label='Predictions', color='orange', linestyle='--', marker='x', markersize=4)
plt.title('Comparison of True Values and Predictions', fontsize=20)
plt.xlabel('Sample Index', fontsize=16)
plt.ylabel('Target Value', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)

# Highlight a specific section for better visibility if necessary
highlight_range = range(50)  # Change this to the range you want to highlight
plt.plot(highlight_range, y_true[highlight_range], color='blue', linestyle='-', marker='o', markersize=6)
plt.plot(highlight_range, y_pred[highlight_range], color='orange', linestyle='--', marker='x', markersize=6)
plt.savefig("pred_transformer_lstm.svg", format='svg')
plt.show()



df = pd.DataFrame(train_losses, columns=["GRU"])
# 保存到 CSV 文件
df.to_csv('loss_GRU.csv', index=False)
# 将 numpy 数组转换为 DataFrame
df = pd.DataFrame(y_pred, columns=["GRU"])

# 保存到 CSV 文件
df.to_csv('pred_GRU.csv', index=False)