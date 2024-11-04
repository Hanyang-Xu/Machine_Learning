import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(-10, 10, 100)
y = np.sin(x) + 0.1 * np.random.randn(100)  # 加入一些噪声
print(x.shape)
print(y.shape)

# 转换为tensor
x_tensor = torch.FloatTensor(x).view(-1, 1)
y_tensor = torch.FloatTensor(y).view(-1, 1)

# 定义神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # 输入层到隐藏层
        self.fc2 = nn.Linear(10, 1)   # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 隐藏层激活
        x = self.fc2(x)               # 输出层
        return x

# 初始化模型、损失函数和优化器
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

# 可视化结果
model.eval()
predicted = model(x_tensor).detach().numpy()
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, predicted, color='red', label='Fitted Curve')
plt.legend()
plt.show()
