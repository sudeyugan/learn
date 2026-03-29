import numpy as np

class ScaledDotProductAttention:
    def __init__(self, d_k):
        """
        d_k: Q 和 K 向量的维度 (用于缩放)
        """
        self.d_k = d_k

    def softmax(self, x, axis=-1):
        # 稳定版 Softmax
        max_x = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def forward(self, Q, K, V, mask=None):
        """
        Q: 形状 (T, d_k)
        K: 形状 (T, d_k)
        V: 形状 (T, d_v) 
        mask: 掩码矩阵，形状 (T, T)，有效位置为 1，PAD 位置为 0
        """
        
        # 1. 计算点积得分
        # TODO: Q 和 K的转置相乘
        scores = Q @ K.T # 替换
        
        # 2. 缩放 (Scale)
        # TODO: 除以 d_k 的平方根 (提示: np.sqrt())
        scores = scores / np.sqrt(self.d_k) # 替换
        
        # 3. 掩码机制 (Masking)
        if mask is not None:
            # TODO: 使用 np.where(条件, 满足条件时的值, 不满足条件时的值)
            # 如果 mask == 0，我们将 scores 对应位置替换为 -1e9，否则保持 scores 不变
            scores = np.where(mask == 0, -1e9, scores)  # 替换
            
        # 4. 计算注意力权重
        # TODO: 传入 self.softmax
        attention_weights = self.softmax(scores, -1) # 替换
        
        # 5. 加权求和得到最终输出 Z
        # TODO: attention_weights 与 V 相乘
        Z = attention_weights @ V # 替换
        
        return Z, attention_weights

# ======= 测试与可视化代码 =======
if __name__ == "__main__":
    np.random.seed(42)
    
    # 假设序列长度 T=4, 维度 d_k=d_v=8
    T, d_k = 4, 8
    
    # 模拟 Q, K, V (这里简单起见直接随机生成，实际中它们是 X 乘以不同的权重矩阵得到的)
    Q = np.random.randn(T, d_k)
    K = np.random.randn(T, d_k)
    V = np.random.randn(T, d_k)
    
    # 模拟掩码机制：假设第 4 个词是 <PAD>
    # 我们希望前 3 个词不要去关注第 4 个词，第 4 个词也不用关注别人。
    # 构建一个 4x4 的 mask，前 3 行前 3 列为 1，涉及到第 4 行/列的都为 0
    mask = np.ones((T, T))
    mask[:, 3] = 0 # 任何词都不能看第 4 个词
    mask[3, :] = 0 # 第 4 个词也不能看任何词 (这一行其实无所谓，因为最终输出也不会用它的结果)
    
    print("输入的掩码矩阵 (Mask):")
    print(mask)
    
    attention = ScaledDotProductAttention(d_k=d_k)
    Z, attn_weights = attention.forward(Q, K, V, mask=mask)
    
    print("-" * 30)
    print("魔法时刻：经过 Softmax 后的注意力权重矩阵 (Attention Weights):")
    print(np.round(attn_weights, 3))
    print(f"\n注意看最后一列！是不是全部变成 0 了？这就是 Mask 的力量！")