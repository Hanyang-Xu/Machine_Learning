import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE

# 加载数据
def load_data(file, ratio, random_state=None):
    dataset = np.load(file, allow_pickle=True)
    dataset = np.array(dataset, dtype=float)
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(dataset.shape[0])
    np.random.shuffle(indices)

    split_index = int(dataset.shape[0] * (1 - ratio))
    train_indices, test_indices = indices[:split_index], indices[split_index:]
    train_data = dataset[train_indices]
    test_data = dataset[test_indices]
    X_train = train_data[:, :6]
    y_train = train_data[:, 6]
    X_test = test_data[:, :6]
    y_test = test_data[:, 6]
    return X_train, X_test, y_train, y_test

# 数据加载
X_train, X_test, y_train, y_test = load_data('midterm_project/ai4i2020.npy', 0.3, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用 SMOTE 对训练数据进行过采样
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 检查 CUDA 是否可用，并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(6, 256)  # 更大规模的第一层
        self.fc2 = nn.Linear(256, 128)  # 更大规模的第二层
        self.fc3 = nn.Linear(128, 2)  # 输出层有 2 个神经元
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 最后一层没有激活函数，CrossEntropyLoss 会自动处理
        return x  # 不再使用 Softmax

# 实例化模型并移动到 GPU/CPU
model = SimpleNN().to(device)

# 定义损失函数（加权的交叉熵损失函数）
class_weights = torch.tensor([1.0, 5.0]).to(device)  # 例：对少数类加大权重
criterion = nn.CrossEntropyLoss(weight=class_weights)  # 加权交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 移动到 CUDA 设备
        optimizer.zero_grad()  # 清空梯度
        outputs = model(inputs)  # 正向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 测试模型
model.eval()
y_pred = []
y_true = []
threshold = 0.6  # 调整阈值

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 移动到 CUDA 设备
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)  # 获取概率分布
        predicted = (probs[:, 1] > threshold).long()  # 应用新的阈值
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# 计算各项指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")

