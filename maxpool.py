import numpy as np

class MaxPooling2D:
    def __init__(self, pool_size=2, stride=2):
        self.F = pool_size # 窗口大小
        self.S = stride    # 滑动步长

    def forward(self, X):
        """
        前向传播：执行 2D 最大池化
        X: 输入的图像矩阵，形状为 (H, W)
        """
        H, W = X.shape
        
        # 1. 计算输出矩阵的形状 (使用带有步长 S 的公式，注意使用 int 确保是整数)
        # TODO: H_out = (H - F) / S + 1
        H_out = (int)((H - self.F) / self.S + 1) # 替换为你的代码
        W_out = (int)((W - self.F) / self.S + 1) # 替换为你的代码
        
        Y = np.zeros((H_out, W_out))
        
        # 2. 执行滑动操作
        for i in range(H_out):
            for j in range(W_out):
                # 3. 计算当前窗口在原图中的确切坐标
                # 因为有步长 S，所以起点不再是 i 和 j，而是 i*S 和 j*S
                # TODO: 计算行起点 r_start, 行终点 r_end, 列起点 c_start, 列终点 c_end
                r_start = i * self.S # 替换
                r_end = i * self.S + self.F   # 替换
                c_start = j * self.S # 替换
                c_end = j * self.S + self.F  # 替换
                
                # 4. 提取当前滑动窗口
                # TODO: 使用计算好的坐标从 X 中切片
                window = X[r_start:r_end, c_start:c_end] # 替换
                
                # 5. 执行最大池化
                # TODO: 取 window 中的最大值赋给 Y[i, j]，提示：使用 np.max()
                Y[i, j] = np.max(window) # 替换
                
        return Y

# ======= 测试代码 =======
if __name__ == "__main__":
    # 构造一个 4x4 输入图像
    X_test = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [0, 1, 8, 9],
        [2, 3, 4, 5]
    ])
    
    # 典型的池化层：2x2窗口，步长为2
    pool = MaxPooling2D(pool_size=2, stride=2)
    
    print("输入的 4x4 图像:")
    print(X_test)
    
    output = pool.forward(X_test)
    
    print("\n最大池化后的输出 (形状应为 2x2):")
    print(output)
    # 预期输出应该是:
    # [[6 4]
    #  [3 9]]