import numpy as np

class Flatten:
    def forward(self, X):
        self.original_shape = X.shape
        X_flat = X.reshape(-1, 1)
        return X_flat

    def backward(self, dX_flat):
        """
        dX_flat: 来自全连接层的梯度，形状为 (C * H * W, 1)
        """
        # 1. 展平层的反向传播
        # TODO: 使用 reshape 将 dX_flat 恢复成前向传播时记录的 original_shape
        dX = dX_flat.reshape(self.original_shape) # 替换为你的代码
        return dX

class MaxPooling2D:
    def __init__(self, pool_size=2, stride=2):
        self.F = pool_size
        self.S = stride

    def forward(self, X):
        self.X = X # 保存输入，反向传播时需要用它来找最大值的位置
        H, W = X.shape
        H_out = (int)((H - self.F) / self.S + 1)
        W_out = (int)((W - self.F) / self.S + 1)
        Y = np.zeros((H_out, W_out))
        
        for i in range(H_out):
            for j in range(W_out):
                r_start = i * self.S
                r_end = i * self.S + self.F
                c_start = j * self.S
                c_end = j * self.S + self.F
                window = X[r_start:r_end, c_start:c_end]
                Y[i, j] = np.max(window)
        return Y

    def backward(self, dY):
        """
        dY: 来自后一层的梯度，形状为 (H_out, W_out)
        """
        # 初始化 dX 为与输入 X 形状相同的全 0 矩阵
        dX = np.zeros_like(self.X)
        H_out, W_out = dY.shape
        
        for i in range(H_out):
            for j in range(W_out):
                # 计算当前窗口的坐标
                r_start = i * self.S
                r_end = i * self.S + self.F
                c_start = j * self.S
                c_end = j * self.S + self.F
                
                # 获取前向传播时的窗口数据
                window = self.X[r_start:r_end, c_start:c_end]
                
                # 2. 核心逻辑：梯度路由 (Gradient Routing)
                # TODO a: 找到 window 中的最大值
                max_val = np.max(window) # 替换
                
                # TODO b: 生成一个掩码 mask，形状和 window 一样。最大值所在位置为 1，其他为 0。
                # 提示：可以直接用 (window == max_val) 进行布尔比较
                mask = (window == max_val) # 替换
                
                # TODO c: 将接收到的梯度 dY[i, j] 乘以 mask，然后累加到 dX 对应的窗口区域中
                dX[r_start:r_end, c_start:c_end] += dY[i, j] * mask # 替换 (0 换成你的代码)
                
        return dX

# ======= 测试代码 =======

if __name__ == "__main__":
    # 1. 测试 Flatten Backward
    flat = Flatten()
    X_test = np.random.randn(2, 3, 3) # 模拟通道为2，3x3 的特征图
    print(f"Flatten 原始输入形状: {X_test.shape}")
    
    out_flat = flat.forward(X_test)
    # 模拟一个来自后方的梯度
    d_out_flat = np.ones_like(out_flat) 
    d_X_test = flat.backward(d_out_flat)
    print(f"Flatten 反向传播输出形状: {d_X_test.shape} (应恢复为 (2, 3, 3))")

    print("-" * 30)

    # 2. 测试 MaxPooling Backward
    pool = MaxPooling2D(pool_size=2, stride=2)
    # 给定一个 4x4 的简单输入
    X_pool_test = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [0, 1, 8, 9],
        [2, 3, 4, 5]
    ])
    print("池化层输入:\n", X_pool_test)
    
    Y_pool_out = pool.forward(X_pool_test)
    # 模拟池化层收到全是 10 的梯度
    dY_pool = np.array([
        [10, 10],
        [10, 10]
    ])
    
    dX_pool = pool.backward(dY_pool)
    print("\n池化层反向传播梯度 (只有前向最大值的位置才有梯度):\n", dX_pool)