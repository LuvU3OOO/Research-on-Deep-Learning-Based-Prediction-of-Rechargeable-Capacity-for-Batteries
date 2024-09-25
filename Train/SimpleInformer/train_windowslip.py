import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from main_Informer import Informer
from Train.utils.window_slip2 import train_loader, test_loader, X_train_tensor, X_test_tensor, data, \
    train_features_tensor, y_train_tensor, test_features_tensor,y_test_tensor
from torchinfo import summary
import shap
# 模型参数
enc_in = 13  # 输入特征的维度
dec_in = enc_in  # 解码器输入的维度与编码器相同
c_out = 1  # 输出特征的维度（对于回归任务通常是 1）
seq_len = 5  # 序列长度为1，因为我们进行单步预测
label_len = 1  # 标签长度，对于单步预测也是1
out_len = 1  # 输出序列的长度，对于单步预测通常是 1
e_layers = 1  # 编码器层数
d_layers = 1  # 解码器层数
hidden_dim = 512  # 前馈网络隐藏层的维度因子
n_heads = 8  # 注意力头的数量
dropout = 0.1  # Dropout概率
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = Informer(enc_in, dec_in, c_out, seq_len, label_len, out_len, e_layers,
                 d_layers, hidden_dim, n_heads, dropout).to(device)
# summary(model, [(32,5,13),(32,5,13),(32,5,5)])
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)

losses = []
# 训练循环
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets, time_f) in enumerate(train_loader):
        # 移动数据到设备
        inputs, targets, time_f = inputs.to(device), targets.to(device), time_f.to(device)
        # print(inputs.shape,targets.shape,time_f.shape)
        optimizer.zero_grad()
        outputs = model(inputs, inputs, time_f)  # 由于单步预测，输入数据用作编码器和解码器的输入

        outputs = outputs.squeeze(-1)  # 去掉最后一个维度，使输出形状变为 [batch_size]
        targets = targets.squeeze(-1)  # 确保目标的形状也是 [batch_size]
        loss = criterion(outputs, targets)  # 损失函数不需要添加额外的维度
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    losses.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 绘制损失曲线
plt.figure(figsize=(12, 6))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Informer Training Loss Curve')
plt.legend()
plt.savefig("Loss_Informer.svg", format='svg')
plt.show()

# 将模型切换到评估模式
model.eval()

# 存储真实值和预测值
true_values = []
predicted_values = []

# 不计算梯度以加快计算速度
with torch.no_grad():
    for inputs, labels, time_f in test_loader:
        # 移动数据到设备
        inputs, labels, time_f = inputs.to(device), labels.to(device), time_f.to(device)
        outputs = model(inputs, inputs, time_f)  # 同样，解码器输入简化处理
        true_values.extend(labels.cpu().numpy())  # 确保从 GPU 移动到 CPU 再转换为 numpy
        y_pred = outputs.squeeze().cpu().numpy()  # 确保从 GPU 移动到 CPU 再转换为 numpy
        predicted_values.extend(y_pred)



# 计算评估指标
y_pred = np.array(predicted_values)
y_true = np.array(true_values)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
r2 = r2_score(y_true, y_pred)

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'R2: {r2:.2f}')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 添加 DejaVu Sans 以支持负号
# Plot comparison of true values and predictions
plt.figure(figsize=(18, 8))
plt.plot(y_true, label='True Values', color='r', linestyle='-', marker='o', markersize=4)
plt.plot(y_pred, label='Predictions', color='y', linestyle='--', marker='x', markersize=4)
plt.title('Informer Comparison of True Values and Predictions', fontsize=20)
plt.ylabel('充电量(Ah)', fontsize=16)
# 设置坐标轴刻度值的字体大小
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.grid(True)
plt.legend()
plt.savefig("prediction_Informer.svg", format='svg')
plt.show()

# 将 numpy 数组转换为 DataFrame
df = pd.DataFrame(losses, columns=["Informer"])
# 保存到 CSV 文件
df.to_csv('loss_Informer.csv', index=False)

# 将 numpy 数组转换为 DataFrame
df = pd.DataFrame(y_pred, columns=["Informer"])

# 保存到 CSV 文件
df.to_csv('pred_Informer.csv', index=False)
# SHAP
background = X_train_tensor[:100].to(device)
b1 = train_features_tensor[:100].to(device)
b2 = y_train_tensor[:100].to(device)
test_data = [X_test_tensor[:100].to(device), X_test_tensor[:100].to(device), test_features_tensor.to(device)]

# Compute SHAP values using GradientExplainer
model.train()
explainer = shap.GradientExplainer(model, [background, background, b1])
model.eval()

# 计算 SHAP 值
shap_values = explainer.shap_values(test_data)

# 只提取 background 对应的 SHAP 值
shap_values_background = shap_values[0]  # 假设背景特征在第一个位置

# 获取 SHAP 值并调整形状
shap_values_flat = shap_values_background.reshape(-1, 13)  # 确保是 (num_samples, 13)

# 获取 X_display
X_display = background.cpu().numpy().reshape(-1, 13)  # 直接获取 background 数据

# 获取特征名称
feature_names = data.columns[:13]  # 确保特征名称与数据一致

# 检查特征和 SHAP 值的维度
print(f"SHAP values shape: {shap_values_flat.shape}, X_display shape: {X_display.shape}")

# # 可视化第一个预测的解释
# shap.initjs()
# force_plot = shap.force_plot(np.mean(shap_values_flat, axis=0), shap_values_flat[0], X_display[0], feature_names=feature_names)
#
# # 保存为 HTML 文件
# shap.save_html("force_plot_0.html", force_plot)

# 绘制 SHAP 摘要条形图
shap.summary_plot(shap_values_flat, X_display, feature_names=feature_names, plot_type="bar", plot_size=(10, 6))
