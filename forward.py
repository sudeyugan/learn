import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size=1):
        # 1. 极其重要的参数初始化！
        # 在深度学习中，权重不能初始化为 0（会导致对称性破坏失败），必须是随机小数。
        # TODO: 使用 np.random.randn(行, 列) * 0.01 来初始化 W1 和 W2
        # TODO: 使用 np.zeros((1, 列)) 初始化 b1 和 b2
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01# 形状 (input_size, hidden_size)
        self.b1 = np.zeros((1,hidden_size)) # 形状 (1, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01# 形状 (hidden_size, output_size)
        self.b2 = np.zeros((1, output_size)) # 形状 (1, output_size)

    def relu(self, Z):
        # 2. 实现 ReLU 激活函数
        # TODO: 返回 max(0, Z)，提示：使用 np.maximum(0, Z)
        return np.maximum(0, Z) # 替换为你的代码

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward(self, X):
        # 3. 第一层：隐藏层前向传播
        # TODO: Z1 = X * W1 + b1
        self.Z1 = X @ self.W1 + self.b1  # 替换为你的代码
        
        # TODO: A1 = relu(Z1)
        self.A1 = self.relu(self.Z1) # 替换为你的代码

        # 4. 第二层：输出层前向传播
        # TODO: Z2 = A1 * W2 + b2
        self.Z2 = self.A1 @ self.W2 + self.b2  # 替换为你的代码
        
        # TODO: A2 = sigmoid(Z2)
        self.A2 = self.sigmoid(self.Z2) # 替换为你的代码
        
        return self.A2

# ======= 测试代码 =======
if __name__ == "__main__":
    np.random.seed(42) # 固定随机种子以保证结果一致
    
    # 假设我们有 5 个样本，每个样本 3 个特征
    X_test = np.random.randn(5, 3) 
    
    # 构建一个神经网络：输入3，隐藏层4个神经元，输出1
    nn = TwoLayerNet(input_size=3, hidden_size=4, output_size=1)
    
    # 运行前向传播
    predictions = nn.forward(X_test)
    
    print("前向传播输出的预测概率 (形状应该是 (5, 1)):")
    print(predictions)
    print(f"实际输出形状: {predictions.shape}")