import numpy as np

class MockPositionalEncoding:
    def forward(self, x):
        # 假装加上了位置编码，直接返回
        return x


class MockEncoderLayer:
    def __init__(self, d_model):
        self.d_model = d_model
    def forward(self, x, mask=None):
        # 假装经过了 MHA, Add&Norm, FFN, Add&Norm
        # 输入是 (T, d_model)，输出必须绝对保持 (T, d_model)
        return x

class Embedding:
    def __init__(self, vocab_size, d_model):
        # 初始化词典的 Embedding 矩阵 (查表矩阵)
        # 形状: (vocab_size, d_model)
        self.W_E = np.random.randn(vocab_size, d_model) * 0.01

    def forward(self, x):
        """
        x: 输入的词索引数组，是一维的整数数组，形状为 (T,)
        """
        # TODO 1: 极其简单的查表操作！利用 NumPy 的高级索引特性
        # 提示: 在 NumPy 中，如果你有一个大矩阵 W_E，你可以直接用数组 x 作为行索引
        # 比如 self.W_E[x] 就能瞬间拔出所有的词向量！
        out = self.W_E[x] # 替换为你的代码
        return out

class TransformerEncoder:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers):
        # 1. 实例化入口组件
        self.embed = Embedding(vocab_size, d_model)
        self.pe = MockPositionalEncoding()
        
        # 2. 疯狂堆叠 N 层 EncoderLayer！
        # TODO 2: 用 Python 的列表推导式，创建一个包含 num_layers 个 MockEncoderLayer 的列表
        # 提示: [MockEncoderLayer(d_model) for _ in range(num_layers)]
        self.layers = [MockEncoderLayer(d_model) for _ in range(num_layers)] # 替换为你的代码

    def forward(self, x, mask=None):
        """
        x: 词索引构成的句子，形状为 (T,)
        """
        # 1. 词嵌入查表：把 (T,) 变成 (T, d_model)
        out = self.embed.forward(x)
        
        # 2. 注入位置编码 (注入时间灵魂)
        out = self.pe.forward(out)
        
        # 3. 穿越 N 层深邃的编码器
        # TODO 3: 写一个极其优雅的 for 循环，让 out 依次穿过 self.layers 里的每一层！
        # 每一层的输出，都要覆盖掉旧的 out，作为下一层的输入。
        for layer in self.layers:
            out = layer.forward(out, mask) # 替换为你的代码        
        return out

# ======= 测试代码 =======
if __name__ == "__main__":
    np.random.seed(42)
    
    # 定义超参数
    vocab_size = 10000  # 字典里有 10000 个词
    d_model = 512       # 词向量维度
    num_layers = 6      # 经典的 6 层堆叠
    T = 8               # 这句话有 8 个词
    
    # 模拟一句话的输入，比如 "我 爱 人 工 智 能 技 术" 对应的索引
    x_input = np.array([5, 89, 230, 11, 45, 9, 1024, 666])
    
    print(f"原始输入的数字序列形状: {x_input.shape}")
    
    # 实例化组装好的庞然大物
    encoder = TransformerEncoder(vocab_size, d_model, num_heads=8, d_ff=2048, num_layers=num_layers)
    
    # 前向传播
    final_output = encoder.forward(x_input)
    
    print(f"经过 Embedding、PE 和 {num_layers} 层 EncoderLayer 后的终极张量形状:")
    print(f"{final_output.shape} (预期绝对必须是 (8, 512))")
    print("\n太伟大了！你已经亲手打造了 ChatGPT 最底层的左半边大脑！")