import pandas as pd
import matplotlib.pyplot as plt

# 从CSV文件读取数据到DataFrame
df = pd.read_csv('plt/loss.csv')

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(df['Informer'], label='Informer')
plt.plot(df['Transformer-LSTM'], label='Transformer-LSTM')
plt.plot(df['Transformer-BiLSTM'], label='Transformer-BiLSTM')
plt.plot(df['Transformer'], label='Transformer')
plt.plot(df['LSTM'], label='LSTM')
plt.plot(df['BiLSTM'], label='BiLSTM')


# plt.plot(df['RNN'], label='RNN')


# 添加图例和标签
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Transformer-LSTM/BiLSTM Train Loss')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.savefig("loss_BI_LSTM.svg", format='svg')
# 显示图表
plt.show()
