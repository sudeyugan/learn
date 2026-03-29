import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.W = None
        self.b = None

    def fit(self, X, y):
        # 确保 y 是一个列向量，形状为 (m, 1)
        y = y.reshape(-1, 1) 
        
        m_samples, n_features = X.shape
        
        # 1. 初始化参数：将 W 初始化为全零的列向量 (n_features, 1)，b 初始化为 0
        # TODO: 使用 np.zeros()
        self.W = np.zeros((n_features,1))
        self.b = 0

        # 梯度下降迭代
        for _ in range(self.n_iters):
            # 2. 前向传播：计算预测值 y_predicted
            # TODO: 根据公式 \hat{Y} = XW + b
            y_predicted = X @ self.W + self.b # 替换为你的代码
            
            # 3. 计算梯度：dw 和 db
            # TODO: 根据公式推导，注意 X 的转置可以使用 X.T
            dw = 1 / m_samples * X.T @ (y_predicted - y) # 替换为你的代码
            db = 1 / m_samples * np.sum(y_predicted - y, axis = 0, keepdims = True) # 替换为你的代码 (提示：使用 np.sum()，别忘了除以 m_samples)
            
            # 4. 参数更新
            # TODO: 使用学习率 self.lr 更新 self.W 和 self.b
            self.W -= self.lr * dw # 替换为你的代码
            self.b -= self.lr * db # 替换为你的代码

    def predict(self, X):
        # 5. 预测新数据
        # TODO: 返回 XW + b
        return X @ self.W + self.b # 替换为你的代码

# ======= 测试代码 (你可以直接运行它来检验) =======
if __name__ == "__main__":
    # 生成一些简单的线性假数据
    X_test = np.array([[1], [2], [3], [4]])
    y_test = np.array([2, 4, 6, 8]) # 关系很明显是 y = 2x

    model = LinearRegression(learning_rate=0.05, n_iters=100)
    model.fit(X_test, y_test)
    
    print(f"训练出的权重 W: \n{model.W}")
    print(f"训练出的偏置 b: {model.b}")
    
    predictions = model.predict(np.array([[5]]))
    print(f"预测输入 5 的结果: {predictions}")