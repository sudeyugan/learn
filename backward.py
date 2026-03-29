import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size=1):
        self.W1 = np.random.randn(input_size, hidden_size) 
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, Z):
        return np.maximum(0, Z)
        
    def relu_derivative(self, Z):
        # 1. ReLU 的导数
        # TODO: 当 Z > 0 时返回 1，否则返回 0。提示：可以直接用 (Z > 0).astype(float)
        return (Z > 0).astype(float) # 替换为你的代码

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)

        # 2. 第二层梯度计算 (输出层)
        # TODO: dZ2 = A2 - y
        dZ2 = self.A2 - y # 替换为你的代码
        
        # TODO: dW2 = (1/m) * A1的转置 @ dZ2
        self.dW2 = (1/m) * self.A1.T @ dZ2 # 替换为你的代码
        self.db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # 3. 第一层梯度计算 (隐藏层)
        # TODO: dA1 = dZ2 @ W2的转置
        dA1 = dZ2 @ self.W2.T  # 替换为你的代码
        
        # TODO: dZ1 = dA1 * relu的导数(Z1)  (注意这里是逐元素乘法 *, 不是 @)
        dZ1 = dA1 * self.relu_derivative(self.Z1) # 替换为你的代码
        
        # TODO: dW1 = (1/m) * X的转置 @ dZ1
        self.dW1 = (1/m) * X.T @ dZ1  # 替换为你的代码
        self.db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

    def update_params(self, learning_rate):
        # 4. 更新参数
        # TODO: 使用 learning_rate 减去对应的梯度 (dW1, db1, dW2, db2)
        self.W1 -= learning_rate * self.dW1 # 替换
        self.b1 -= learning_rate * self.db1 # 替换
        self.W2 -= learning_rate * self.dW2 # 替换
        self.b2 -= learning_rate * self.db2 # 替换

# ======= 测试与训练代码 =======
if __name__ == "__main__":
    np.random.seed(42)
    # 我们来做一个异或(XOR)类型的非线性数据集，逻辑回归无法解决这个问题！
    X_train = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_train = np.array([0, 1, 1, 0]) # 相同为0，不同为1

    # 输入2个特征，隐藏层设为4个神经元，输出1个预测
    nn = TwoLayerNet(input_size=2, hidden_size=4, output_size=1)
    
    epochs = 5000
    learning_rate = 0.5
    
    print("开始训练...")
    for i in range(epochs):
        # 1. 前向传播
        predictions = nn.forward(X_train)
        
        # 2. 计算损失 (BCE)
        loss = -np.mean(y_train.reshape(-1,1) * np.log(predictions + 1e-8) + 
                       (1 - y_train.reshape(-1,1)) * np.log(1 - predictions + 1e-8))
        
        # 3. 反向传播
        nn.backward(X_train, y_train)
        
        # 4. 更新参数
        nn.update_params(learning_rate)
        
        if i % 1000 == 0:
            print(f"Epoch {i}, Loss: {loss:.4f}")

    print("\n训练结束！测试预测结果 (应该接近 [0, 1, 1, 0]):")
    print(nn.forward(X_train))