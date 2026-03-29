import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param # 正则化参数
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # 1. 确保标签 y 是 {-1, 1} 而不是 {0, 1}
        # TODO: 使用 np.where 将 y 中 <= 0 的值替换为 -1，> 0 的替换为 1
        y_ = np.where(y <= 0, -1, 1) # 替换为你的代码
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features) # 注意：这里为了方便迭代，设为一维数组 (n_features,)
        self.b = 0

        # 梯度下降
        for _ in range(self.n_iters):
            # 遍历每一个样本进行随机梯度下降 (SGD)
            for idx, x_i in enumerate(X):
                # 2. 计算条件判断：y_i * (w^T * x_i - b) >= 1
                # TODO: 使用 np.dot 计算 x_i 和 self.w 的点积
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    # 3. 满足条件（分类正确且间隔够大）：只对正则化项求导
                    # TODO: dw = lambda * w, db = 0
                    self.w -= self.lr * (2 * self.lambda_param * self.w) # 这里的2是为了抵消1/2
                    # b 不更新
                else:
                    # 4. 不满足条件（分错或间隔太小）：对正则化项和 Hinge Loss 都求导
                    # TODO: 更新 w 和 b，公式见上方推导的第2种情况
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * (y_[idx]) # 提示：我们公式里设的是 -b

    def predict(self, X):
        # 5. 预测：计算 w^T * X - b，然后取符号（正数返回1，负数返回-1）
        # TODO: 使用 np.dot 计算线性输出，并使用 np.sign() 返回符号
        approx = X @ self.w  - self.b # 替换为你的代码
        return np.sign(approx)

# ======= 测试代码 =======
if __name__ == "__main__":
    # 生成假数据
    X_test = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    y_test = np.array([-1, -1, 1, 1, -1, 1])

    clf = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    clf.fit(X_test, y_test)
    
    test_samples = np.array([[0, 0], [10, 10]])
    predictions = clf.predict(test_samples)
    print(f"输入 {test_samples.tolist()} 的分类预测结果: {predictions.tolist()}")