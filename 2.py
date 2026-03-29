import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.W = None
        self.b = None

    def _sigmoid(self, z):
        # 1. 实现 Sigmoid 函数
        # TODO: 根据公式 1 / (1 + e^(-z))，使用 np.exp()
        return 1 / (1 + np.exp(-z)) # 替换为你的代码

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        m_samples, n_features = X.shape
        
        # 初始化参数 (与线性回归相同)
        self.W = np.zeros((n_features, 1))
        self.b = 0

        for _ in range(self.n_iters):
            # 2. 前向传播：计算线性输出，并套上 sigmoid
            # TODO: 先计算线性部分 Z = XW + b，再计算 y_predicted = sigmoid(Z)
            linear_model = X @ self.W + self.b # 替换为你的代码
            y_predicted = self._sigmoid(linear_model)  # 替换为你的代码
            
            # 3. 计算梯度 (形式与线性回归完全一致！)
            # TODO: 计算 dw 和 db
            dw = 1 / m_samples * X.T @ (y_predicted - y) # 替换为你的代码
            db = 1 / m_samples * np.sum(y_predicted - y, axis = 0, keepdims = True) # 替换为你的代码
            
            # 4. 更新参数
            # TODO: 使用学习率和梯度更新 W 和 b
            self.W -= self.lr * dw # 替换为你的代码
            self.b -= self.lr * db # 替换为你的代码

    def predict(self, X):
        # 5. 预测分类类别
        # 先计算概率，如果概率 >= 0.5 返回 1，否则返回 0
        linear_model = X @ self.W + self.b
        y_predicted = self._sigmoid(linear_model)
        
        # TODO: 使用条件判断将概率转换为 0 或 1，提示：你可以使用 np.where(condition, x, y)
        y_predicted_cls = np.where(y_predicted > 0.5, 1, 0) # 替换为你的代码
        return y_predicted_cls

# ======= 测试代码 =======
if __name__ == "__main__":
    # 生成假数据：简单的二分类任务
    X_test = np.array([[1, 2], [2, 3], [3, 4], [8, 9], [9, 10], [10, 11]])
    y_test = np.array([0, 0, 0, 1, 1, 1]) # 前三个为类0，后三个为类1

    model = LogisticRegression(learning_rate=0.1, n_iters=1000)
    model.fit(X_test, y_test)
    
    # 预测一个介于两类之间的新数据
    test_samples = np.array([[5, 6], [7, 8]])
    predictions = model.predict(test_samples)
    print(f"输入 {test_samples.tolist()} 的分类预测结果: {predictions.flatten().tolist()}")