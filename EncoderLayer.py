import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        # 初始化可学习参数 gamma (全 1) 和 beta (全 0)
        self.gamma = np.ones((d_model,))
        self.beta = np.zeros((d_model,))

    def forward(self, x):
        """ x 的形状通常是 (T, d_model) """
        # TODO 1: 沿着词向量维度 (axis=-1) 计算均值 mean 和方差 var，记得 keepdims=True
        mean = np.mean(x, axis=-1, keepdims=True) # 替换
        var = np.var(x, axis=-1, keepdims=True) # 替换
        
        # TODO 2: 标准化
        x_norm = (x - mean) / np.sqrt(var + self.eps) # 替换 (提示: (x - mean) / np.sqrt(var + self.eps))
        
        # TODO 3: 乘以 gamma 加上 beta
        out = x_norm * self.gamma + self.beta # 替换
        return out

class FeedForward:
    def __init__(self, d_model, d_ff):
        # 初始化两层线性网络的权重
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros((d_ff,))
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros((d_model,))

    def forward(self, x):
        # TODO 4: 第一层线性变换 (x @ W1 + b1)
        hidden = x @ self.W1 + self.b1 # 替换
        
        # TODO 5: ReLU 激活 ( np.maximum(0, hidden) )
        activated = np.maximum(0, hidden) # 替换
        
        # TODO 6: 第二层线性变换 (activated @ W2 + b2)
        out = activated @ self.W2 + self.b2 # 替换
        return out

class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        # 我们这里假设 MultiHeadAttention 已经完美写好了
        # self.mha = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None, mha_output=None):
        """ 
        为了测试方便，我们假设 mha_output 是上一关 MultiHeadAttention 算出来的结果 
        在真实的组合中，这里应该是 mha_output = self.mha.forward(x, mask)
        """
        # 1. 第一次 Add & Norm (针对多头注意力)
        # TODO 7: 将 MHA 的输出与输入 x 相加 (残差连接)
        x_add_1 = x + mha_output # 替换
        
        # TODO 8: 过第一个 LayerNorm
        out_1 = self.norm1.forward(x_add_1)  # 替换
        
        # 2. 第二次 Add & Norm (针对 FFN)
        # TODO 9: 将 out_1 放入 self.ffn 计算
        ffn_output = self.ffn.forward(out_1)  # 替换

        # TODO 10: 残差连接 (ffn_output + out_1)
        x_add_2 = ffn_output + out_1 # 替换
        
        # TODO 11: 过第二个 LayerNorm 得到最终结果
        out_2 = self.norm2.forward(x_add_2) # 替换

        return out_2

# ======= 测试代码 =======
if __name__ == "__main__":
    np.random.seed(42)
    T, d_model, d_ff = 4, 512, 2048
    
    # 模拟进入 EncoderLayer 的张量
    x_input = np.random.randn(T, d_model)
    # 模拟 MHA 的输出张量 (保证形状相同)
    mha_out = np.random.randn(T, d_model)
    
    encoder_layer = EncoderLayer(d_model=d_model, num_heads=8, d_ff=d_ff)
    
    final_output = encoder_layer.forward(x_input, mha_output=mha_out)
    
    print(f"输入形状: {x_input.shape}")
    print(f"经过一层完整 EncoderLayer 后的输出形状: {final_output.shape} (预期绝对必须是 (4, 512))")
    print("太棒了！输入和输出的维度严丝合缝地对齐了！")