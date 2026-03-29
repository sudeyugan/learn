import numpy as np

class Conv2D_Multi:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.C_in = in_channels
        self.C_out = out_channels
        self.F = kernel_size
        
        # 1. 初始化 4D 权重张量和偏置
        # TODO: 使用 np.random.randn 初始化 W，形状为 (C_out, C_in, F, F)，并乘以 0.01
        self.W = np.random.randn(self.C_out, self.C_in, self.F, self.F) * 0.01# 替换为你的代码
        
        # TODO: 初始化偏置 b，形状为 (C_out, 1)
        self.b = np.zeros((self.C_out, 1)) # 替换为你的代码

    def forward(self, X):
        """
        X: 输入图像，形状为 (C_in, H, W)
        """
        C_in, H, W = X.shape
        
        # 2. 计算输出的空间尺寸
        H_out = H - self.F + 1
        W_out = W - self.F + 1
        
        # 3. 初始化输出张量 Y，形状为 (C_out, H_out, W_out)
        Y = np.zeros((self.C_out, H_out, W_out))
        
        # 4. 执行多通道多核卷积
        # 遍历每一个输出通道 (即遍历每一个卷积核)
        for c_out in range(self.C_out):
            # 遍历输出图像的空间位置
            for i in range(H_out):
                for j in range(W_out):
                    
                    # 5. 提取当前的 3D 滑动窗口 (注意这里要包含所有输入通道)
                    # TODO: 行范围是 i 到 i+F, 列范围是 j 到 j+F。通道维度取全部 (可以使用 : )
                    window = X[:, i:i+self.F, j:j+self.F] # 替换为你的代码 (检查提示是否正确)
                    
                    # 6. 计算卷积：将窗口与当前对应的卷积核进行逐元素相乘后求和
                    # TODO: 取出第 c_out 个卷积核 (self.W[c_out])，与 window 逐元素相乘 (*)
                    # 然后使用 np.sum() 把结果加起来，最后加上对应的偏置 self.b[c_out]
                    Y[c_out, i, j] = np.sum(self.W[c_out] * window) + self.b[c_out, 0] # 替换为你的代码
                    
        return Y

class Flatten:
    def forward(self, X):
        """
        X: 卷积/池化层的输出，形状为 (C, H, W)
        """
        self.original_shape = X.shape # 保存原始形状，反向传播时会用到！
        
        # 7. 将 3D 张量拉平为 2D 矩阵 (其实也就是一个列向量)
        # TODO: 使用 reshape 将 X 变成 (C * H * W, 1) 的形状。
        # 提示：你可以用 np.prod(X.shape) 来快速计算总元素个数，或者用 reshape(-1, 1)
        X_flat = X.reshape(-1, 1) # 替换为你的代码
        return X_flat

# ======= 测试代码 =======
if __name__ == "__main__":
    np.random.seed(42)
    
    # 模拟一张 3通道 的 5x5 彩色图像 (比如 RGB)
    X_rgb = np.random.randn(3, 5, 5)
    
    # 构建一个卷积层：输入 3 通道，输出 4 通道 (即 4 个卷积核)，核大小 3x3
    conv = Conv2D_Multi(in_channels=3, out_channels=4, kernel_size=3)
    
    # 构建一个展平层
    flatten = Flatten()
    
    print(f"原始 RGB 图像形状: {X_rgb.shape}")
    
    # 1. 经过多核卷积层
    conv_out = conv.forward(X_rgb)
    print(f"经过卷积层后的形状: {conv_out.shape}  (预期: (4, 3, 3))")
    
    # 2. 经过展平层
    flat_out = flatten.forward(conv_out)
    print(f"经过展平层后的形状: {flat_out.shape}  (预期: (36, 1))")