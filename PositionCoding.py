import numpy as np
import matplotlib.pyplot as plt

class PositionalEncoding:
    def __init__(self, d_model, max_len=100):
        """
        d_model: 词向量的维度
        max_len: 模型能处理的最长序列长度 (提前把所有位置算好备用)
        """
        self.d_model = d_model
        
        # 1. 初始化一个全为 0 的 PE 矩阵，形状为 (max_len, d_model)
        self.pe = np.zeros((max_len, d_model))
        
        # 2. 生成位置索引 pos，形状为 (max_len, 1)
        # 即变成一列: [[0], [1], [2], ..., [max_len-1]]
        position = np.arange(0, max_len).reshape(-1, 1)
        
        # 3. 计算公式中的分母部分：10000^(2i / d_model)
        # 技巧：在数学上，10000^(2i/d_model) 等价于 exp(2i * (-log(10000) / d_model))
        # 我们先生成 2i 的数组：[0, 2, 4, ..., d_model-2]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        # 4. 魔法时刻：利用广播机制计算正弦和余弦！
        # TODO: 给 PE 矩阵的所有偶数列 (0, 2, 4...) 赋值，使用 np.sin()
        # 提示: position * div_term 会自动广播成一个 (max_len, d_model/2) 的矩阵
        self.pe[:, 0::2] = np.sin(position * div_term) # 替换为你的代码
        
        # TODO: 给 PE 矩阵的所有奇数列 (1, 3, 5...) 赋值，使用 np.cos()
        self.pe[:, 1::2] = np.cos(position * div_term) # 替换为你的代码

    def forward(self, X):
        """
        X: 经过 Embedding 后的词向量矩阵，形状为 (T, d_model)
        """
        T = X.shape[0]
        # TODO: 从预先计算好的 self.pe 中，取出前 T 行，直接加到 X 上！
        X_out = X + self.pe[0:T] # 替换为你的代码
        return X_out

# ======= 测试与可视化代码 =======
if __name__ == "__main__":
    d_model = 128
    T = 50 # 我们模拟一个长度为 50 的句子
    
    # 模拟一个全零的 Embedding 输出 (方便我们直接观察 PE 的模样)
    X_embed = np.zeros((T, d_model))
    
    pe_layer = PositionalEncoding(d_model=d_model, max_len=100)
    
    # 前向传播：把位置编码加到词向量上
    X_final = pe_layer.forward(X_embed)
    
    print(f"最终输入矩阵的形状: {X_final.shape} (预期为 (50, 128))")
    
    # 打印前两行，感受一下数值的变化
    print("\n第一个词 (pos=0) 的前 4 个维度的 PE 值:", X_final[0, :4])
    print("第二个词 (pos=1) 的前 4 个维度的 PE 值:", X_final[1, :4])