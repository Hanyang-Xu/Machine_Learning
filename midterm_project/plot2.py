import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 数据
data = {
    'n_iter': [2000, 2000, 2000, 2000, 6000, 6000, 6000, 6000],
    'learning_rate': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    'batch_size': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
    'layers': ["[64, 128, 32, 1]", "[64, 128, 32, 1]", "[64, 128, 32, 1]", "[64, 128, 32, 1]", 
               "[64, 128, 32, 1]", "[64, 128, 32, 1]", "[64, 128, 32, 1]", "[64, 128, 32, 1]"],
    'w0': [1, 1, 1, 1, 1, 1, 1, 1],
    'w1': [1, 30, 1, 30, 1, 30, 1, 30],
    'gate': [0.3, 0.3, 0.5, 0.5, 0.3, 0.3, 0.5, 0.5],
    'precision': [0.6493506493506493, 0.6091954022988506, 0.6268656716417911, 0.8717948717948718,
                  0.6111111111111112, 0.532258064516129, 0.65625, 0.7307692307692307],
    'recall': [0.5263157894736842, 0.5578947368421052, 0.4421052631578947, 0.35789473684210527,
               0.6947368421052632, 0.6947368421052632, 0.6631578947368421, 0.6],
    'f1_score': [0.5813953488372093, 0.5824175824175823, 0.5185185185185185, 0.5074626865671642,
                 0.6502463054187193, 0.6027397260273973, 0.6596858638743456, 0.6589595375722542]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 计算 w1 / w0 的比值
df['w1/w0'] = df['w1'] / df['w0']

# 绘制 f1_score 和 w1/w0 比值的柱状图
plt.figure(figsize=(6, 6))

sns.barplot(x='w1/w0', y='f1_score', data=df, palette='viridis', width=0.6)

# 添加标题和标签
plt.title('F1 Score vs w1/w0', fontsize=16)
plt.xlabel('w1/w0', fontsize=14)
plt.ylabel('F1 Score', fontsize=14)
plt.savefig('w1_w0.png', transparent=True)
# 显示图形
plt.tight_layout()
plt.show()
