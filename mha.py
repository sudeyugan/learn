import numpy as np

class ScaledDotProductAttention:
    def __init__(self, d_k):
        self.d_k = d_k

    def softmax(self, x, axis=-1):
        max_x = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def forward(self, Q, K, V, mask=None):
        scores = Q @ K.swapaxes(-1, -2) # 注意：这里用 swapaxes 保证只转置最后两维！
        scores = scores / np.sqrt(self.d_k)
        if mask is not None:
            # mask 可能需要广播，以适应多头维度
            scores = np.where(mask == 0, -1e9, scores) 
        attention_weights = self.softmax(scores, -1)
        Z = attention_weights @ V
        return Z, attention_weights

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_k = d_model // num_heads
        
        # 1. 初始化巨大的合并权重矩阵
        self.W_Q = np.random.randn(d_model, d_model) * 0.01
        self.W_K = np.random.randn(d_model, d_model) * 0.01
        self.W_V = np.random.randn(d_model, d_model) * 0.01
        self.W_O = np.random.randn(d_model, d_model) * 0.01
        
        # 实例化内部的单头注意力计算模块
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, X, mask=None):
        """
        X: 形状为 (T, d_model)
        mask: 形状为 (T, T) 的二维掩码矩阵 (可选)
        """
        T = X.shape[0]
        
        # 2. 一次性映射得到全体 Q, K, V (形状均为 (T, d_model))
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V
        
        # 3. 核心魔法：变形与多头切分 (Reshape and Transpose)
        # TODO a: 将 Q, K, V reshape 为 (T, num_heads, d_k)
        Q_split = Q.reshape(T, self.num_heads, self.d_k) # 替换
        K_split = K.reshape(T, self.num_heads, self.d_k) # 替换
        V_split = V.reshape(T, self.num_heads, self.d_k) # 替换
        
        # TODO b: 交换前两个维度，使形状变为 (num_heads, T, d_k)
        # 提示：使用 np.swapaxes(矩阵, 0, 1) 或者 .transpose(1, 0, 2)
        Q_heads = np.swapaxes(Q_split, 0, 1) # 替换
        K_heads = np.swapaxes(K_split, 0, 1) # 替换
        V_heads = np.swapaxes(V_split, 0, 1) # 替换
        
        # 处理掩码的维度广播：如果传了二维的 (T, T) 掩码，我们需要给它增加一个头的维度
        # 变成 (1, T, T)，这样 numpy 就能自动把它广播到 (num_heads, T, T) 上去
        if mask is not None:
            mask[np.newaxis, ...]

        # 4. 投入平行宇宙计算注意力
        # 注意：这里的输入都是三维的 (num_heads, T, d_k)，我们的 ScaledDotProductAttention 完美支持三维矩阵乘法！
        Z_heads, _ = self.attention.forward(Q_heads, K_heads, V_heads, mask)
        # 此时 Z_heads 的形状应该是 (num_heads, T, d_k)
        
        # 5. 收束时间线：拼接多头并输出
        # TODO c: 交换回前两个维度，使形状变回 (T, num_heads, d_k)
        Z_split_back = np.swapaxes(Z_heads, 0, 1) # 替换
        
        # TODO d: 将三维矩阵强行拉平回二维 (T, d_model)
        # 提示：使用 reshape(T, self.d_model)
        Z_concat = Z_split_back.reshape(T, self.d_model) # 替换
        
        # TODO e: 最后乘上输出融合矩阵 W_O
        output = Z_concat @ self.W_O # 替换
        
        return output

# ======= 测试代码 =======
if __name__ == "__main__":
    np.random.seed(42)
    T = 4           # 序列长度
    d_model = 512   # 模型维度
    num_heads = 8   # 8 个平行宇宙 (头)
    
    # 模拟输入序列
    X_sentence = np.random.randn(T, d_model)
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    output = mha.forward(X_sentence)
    
    print(f"输入形状: {X_sentence.shape}")
    print(f"经过多头注意力机制后的输出形状: {output.shape} (预期绝对必须是 (4, 512))")