import numpy as np

# ====== 模拟你之前写好的完美组件 ======
class MockMHA:
    def forward(self, Q, K, V, mask=None):
        return Q # 模拟输出形状与 Q 一致

class LayerNorm:
    def forward(self, x):
        return x

class FeedForward:
    def forward(self, x):
        return x
# ======================================

class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        # 1. 掩码多头自注意力 (Masked Self-Attention)
        self.masked_mha = MockMHA()
        self.norm1 = LayerNorm()
        
        # 2. 交叉多头注意力 (Cross-Attention)
        self.cross_mha = MockMHA()
        self.norm2 = LayerNorm()
        
        # 3. 前馈神经网络 (FFN)
        self.ffn = FeedForward()
        self.norm3 = LayerNorm()

    def forward(self, dec_input, enc_output, causal_mask=None, cross_mask=None):
        """
        dec_input: Decoder 当前的输入 (T_dec, d_model)
        enc_output: Encoder 的终极输出 (T_enc, d_model)
        causal_mask: 因果掩码 (T_dec, T_dec)
        cross_mask: 交叉掩码，防止关注 Encoder 端的 PAD (T_dec, T_enc)
        """
        # TODO 1: 实现带有残差连接和 LayerNorm 的 Masked Self-Attention
        attn1 = self.masked_mha.forward(dec_input, dec_input, dec_input, causal_mask)
        out1 = self.norm1.forward(dec_input + attn1)

        # TODO 2: 实现带有残差连接和 LayerNorm 的 Cross-Attention
        # 极度危险警告：想清楚 Q, K, V 分别是谁传入的！
        attn2 = out1 + self.cross_mha.forward(out1, enc_output, enc_output, cross_mask)
        out2 = self.norm1.forward(out1 + attn2)
        
        # TODO 3: 实现带有残差连接和 LayerNorm 的 FFN
        ffn_out = self.ffn.forward(out2)
        out3 = self.norm3.forward(out2 + ffn_out)
        
        return out3 # 替换为你最终的变量名




def get_causal_mask(seq_len):
    """
    生成因果掩码 (Causal Mask)
    形状: (seq_len, seq_len)
    要求: 对角线及以下为 1，以上为 0
    """
    # TODO 4: 使用 NumPy 生成这个极其重要的下三角矩阵
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask


# ======= 测试代码 =======
if __name__ == "__main__":
    T_enc = 10      # 源语言序列长度
    T_dec = 5       # 目标语言序列长度
    d_model = 512
    
    # 模拟张量
    enc_out = np.random.randn(T_enc, d_model)
    dec_in = np.random.randn(T_dec, d_model)
    
    # 1. 测试因果掩码生成
    causal_mask = get_causal_mask(T_dec)
    print("生成的神圣因果掩码 (Causal Mask):")
    print(causal_mask)
    assert causal_mask[0, 1] == 0 and causal_mask[4, 0] == 1, "因果掩码方向反了或生成错误！"
    
    # 2. 测试 DecoderLayer 前向传播
    decoder_layer = DecoderLayer(d_model=d_model, num_heads=8, d_ff=2048)
    final_output = decoder_layer.forward(dec_in, enc_out, causal_mask=causal_mask)
    
    print(f"\n跨越维度的握手！最终 Decoder 输出形状: {final_output.shape}")
    print("预期绝对必须是 (5, 512)")