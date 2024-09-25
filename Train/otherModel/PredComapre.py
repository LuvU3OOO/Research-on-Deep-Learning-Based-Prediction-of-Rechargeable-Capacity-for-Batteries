import pandas as pd
import matplotlib.pyplot as plt

# 从CSV文件读取数据到DataFrame
df = pd.read_csv('plt/true.csv')

# 绘制折线图
plt.figure(figsize=(18, 8))
# plt.plot(df['Informer'],'-',linewidth=1, label='Informer')
plt.plot(df['Transformer-LSTM'], '-.' ,linewidth=1, label='Transformer-LSTM')
plt.plot(df['Transformer-BiLSTM'], '-.' ,linewidth=1, label='Transformer-BiLSTM')
plt.plot(df['Transformer'],'-.',linewidth=1, label='Transformer')
plt.plot(df['LSTM'], '-.', linewidth=1, label='LSTM')
plt.plot(df['BiLSTM'],'-.', linewidth=1, label='BiLSTM')
plt.plot(df['GRU'], linewidth=1, label='GRU')
# plt.plot(df['RNN'], linewidth=1, label='RNN')
plt.plot(df['TRUE'], 'r', marker='o',markersize=3,linewidth=1, label='True Values')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 添加 DejaVu Sans 以支持负号
# 添加图例和标签
plt.legend()
plt.ylabel('充电量(Ah)', fontsize=16)
plt.title('Comparison of True Values and Predictions')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)

# 保存图表
plt.savefig("pred_all.svg", format='svg')

# 显示图表
plt.show()