import numpy as np

class Conv2D:
    def __init__(self, kernel_size):
        # 初始化一个 F x F 的随机卷积核
        self.F = kernel_size
        self.K = np.random.randn(self.F, self.F)

    def forward(self, X):
        """
        前向传播：执行 2D 卷积
        X: 输入的单通道图像矩阵，形状为 (H, W)
        """
        H, W = X.shape
        
        # 1. 计算输出矩阵的形状
        # TODO: 根据公式 H_out = H - F + 1 计算
        H_out = H - self.F + 1 # 替换为你的代码
        W_out = W - self.F + 1 # 替换为你的代码
        
        # 初始化输出矩阵为全 0
        Y = np.zeros((H_out, W_out))
        
        # 2. 执行卷积滑动操作 (双重循环)
        for i in range(H_out):
            for j in range(W_out):
                # 3. 提取当前滑动窗口 (感受野)
                # TODO: 使用切片从 X 中取出大小为 F x F 的子矩阵
                # 提示：行范围是 i 到 i+F，列范围是 j 到 j+F
                window = X[i:i+self.F, j:j+self.F] # 替换为你的代码
                
                # 4. 计算卷积结果
                # TODO: 将 window 和卷积核 self.K 进行逐元素相乘，然后使用 np.sum() 求和
                # 提示：逐元素相乘在 NumPy 中可以直接用 * 符号
                Y[i, j] = np.sum(window * self.K) # 替换为你的代码
                
        return Y

# ======= 测试代码 =======
if __name__ == "__main__":
    np.random.seed(42) # 固定随机种子
    
    # 构造一个简单的 5x5 输入图像
    X_test = np.array([
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0]
    ])
    
    # 初始化一个 3x3 的卷积层
    conv = Conv2D(kernel_size=3)
    
    # 为了方便测试，我们手动设置一个用于提取“对角线边缘”的卷积核
    conv.K = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    print("输入的 5x5 图像:")
    print(X_test)
    print("\n使用的 3x3 卷积核:")
    print(conv.K)
    
    output = conv.forward(X_test)
    
    print(f"\n卷积运算的输出 (形状应为 {(X_test.shape[0] - 3 + 1, X_test.shape[1] - 3 + 1)}):")
    print(output)