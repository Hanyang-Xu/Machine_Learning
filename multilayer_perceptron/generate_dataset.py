import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 生成数据集
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# 将标签从 0 和 1 转换为 0 和 1
y = y + 1  # 保持为 0 和 1，不需要再做 +1 操作

# 绘制数据集
plt.figure(figsize=(8, 8))
plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='green', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Moon Shape Dataset with Labels 0 and 1')
plt.show()

# 将数据集组合到一起
X_ = np.empty([X.shape[0], X.shape[1] + 1])
X_[:, 0] = y  # 标签作为第一列
X_[:, 1:] = X  # 特征数据在剩余列

# 保存到文件
np.savetxt('multilayer_perceptron/X_data.txt', X_, delimiter=',')


