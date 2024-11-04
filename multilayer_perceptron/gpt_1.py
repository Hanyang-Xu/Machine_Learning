import numpy as np

class MultilayerPerceptron:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # 初始化权重和偏置
        for i in range(len(layers) - 1):
            weight = np.random.randn(layers[i], layers[i + 1]) * 0.01
            bias = np.zeros((1, layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):
        self.a = [X]
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(self.a[-1], weight) + bias
            a = self.sigmoid(z)
            self.a.append(a)
        return self.a[-1]

    def backward(self, X, y):
        m = X.shape[0]
        delta = self.a[-1] - y
        for i in reversed(range(len(self.weights))):
            gradient_weight = np.dot(self.a[i].T, delta) / m
            gradient_bias = np.sum(delta, axis=0, keepdims=True) / m
            self.weights[i] -= self.learning_rate * gradient_weight
            self.biases[i] -= self.learning_rate * gradient_bias
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.a[i])

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            if epoch % 100 == 0:
                loss = self.loss(y, self.a[-1])
                print(f'Epoch {epoch}, Loss: {loss}')

    def loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-10))

# 示例使用
if __name__ == "__main__":
    # 创建一个简单的二分类数据集
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR 问题

    # 创建MLP模型
    mlp = MultilayerPerceptron(layers=[2, 2, 1], learning_rate=0.1)

    # 训练模型
    mlp.train(X, y, epochs=1000)

    # 测试模型
    predictions = mlp.forward(X)
    print("Predictions:")
    print(predictions)

