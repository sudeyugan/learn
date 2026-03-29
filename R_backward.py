import numpy as np

class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.d_in = input_dim
        self.d_h = hidden_dim
        self.d_out = output_dim
        
        # 1. 严格按照维度推演初始化权重 (使用极小随机数)
        # TODO: W_xh 负责把 d_in 映射到 d_h
        self.W_xh = np.random.randn(self.d_h, self.d_in) * 0.01 # 替换为 np.random.randn(...) * 0.01
        
        # TODO: W_hh 负责把 d_h 映射到 d_h
        self.W_hh = np.random.randn(self.d_h, self.d_h) * 0.01 # 替换
        
        # TODO: W_hy 负责把 d_h 映射到 d_out
        self.W_hy = np.random.randn(self.d_out, self.d_h) * 0.01 # 替换

        # 初始化偏置
        self.b_h = np.zeros((self.d_h, 1))
        self.b_y = np.zeros((self.d_out, 1))

    def forward(self, X_seq):
        """
        X_seq: 整个时间序列的输入，形状为 (T, d_in)
               其中 T 是时间步的总数 (序列长度)
        """
        T, d_in = X_seq.shape
        
        # 2. 初始化时刻 t=0 的隐藏状态 h_0 (全零列向量)
        # TODO: 形状应为 (d_h, 1)
        h_prev = np.zeros((self.d_h, 1)) # 替换
        
        # 准备缓存，这在以后写反向传播 BPTT 时极其重要！
        self.cache = {'x': [], 'h': [h_prev], 'y': []}
        
        # 3. 时间展开 (Unroll)
        for t in range(T):
            # 取出当前时刻的输入，并转为列向量 (d_in, 1)
            x_t = X_seq[t].reshape(-1, 1)
            
            # 4. 核心：计算当前时刻的隐藏状态 h_t
            # TODO: 根据公式 h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
            h_t = np.tanh(self.W_hh @ h_prev + self.W_xh @ x_t + self.b_h) # 替换为你的代码
            
            # 5. 计算当前时刻的输出 y_t
            # TODO: 根据公式 y_t = W_hy @ h_t + b_y
            y_t = self.W_hy @ h_t + self.b_y # 替换为你的代码
            
            # 记录缓存
            self.cache['x'].append(x_t)
            self.cache['h'].append(h_t)
            self.cache['y'].append(y_t)
            
            # 将当前的记忆传给下一个时间步
            h_prev = h_t
            
        # 返回所有时间步的输出
        return self.cache['y'], self.cache['h']

def backward(self, dy_seq):
        """
        dy_seq: 所有时间步的输出误差，形状为 (T, d_out, 1)
        """
        T = len(dy_seq)
        
        # 1. 初始化所有参数的梯度矩阵为 0
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        # 初始化未来传回来的误差 (最后一步的未来误差是 0)
        dh_next = np.zeros((self.d_h, 1))
        
        # 2. 逆着时间轴循环 (从 T-1 到 0)
        for t in reversed(range(T)):
            # 取出前向传播时的缓存
            dy_t = dy_seq[t]
            x_t = self.cache['x'][t]
            h_t = self.cache['h'][t+1] # 注意：h_0 在索引0，所以 h_t 在 t+1
            h_prev = self.cache['h'][t]
            
            # 3. 计算输出层梯度 (记得是累加！使用 +=)
            # TODO: dW_hy += dy_t @ h_t 的转置
            dW_hy += dy_t @ h_t.T # 替换
            db_y += dy_t
            
            # 4. 计算隐藏层接收到的总误差 dh_t
            # TODO: dh_t = W_hy的转置 @ dy_t + dh_next
            dh_t = self.W_hy.T @ dy_t + dh_next # 替换
            
            # 5. 穿透 tanh 激活函数计算 dz_t
            # TODO: dz_t = dh_t * (1 - h_t的平方)  (注意是逐元素乘法)
            dz_t = dh_t * (1 - h_t * h_t) # 替换
            
            # 6. 计算隐藏层和输入层梯度 (累加！)
            # TODO: dW_hh += dz_t @ h_prev 的转置
            dW_hh += dz_t @ h_prev.T # 替换
            # TODO: dW_xh += dz_t @ x_t 的转置
            dW_xh += dz_t @ x_t.T # 替换
            db_h += dz_t
            
            # 7. 更新传给上一个时刻的误差 dh_next
            # TODO: dh_next = W_hh的转置 @ dz_t
            dh_next = self.W_hh.T @ dz_t # 替换
            
        return dW_xh, dW_hh, dW_hy, db_h, db_y

# ======= 测试代码 =======
if __name__ == "__main__":
    np.random.seed(42)
    
    T = 5          # 序列长度为 5
    input_dim = 10 # 比如 10 维的词向量
    hidden_dim = 6 # 隐藏层大脑容量为 6 维
    output_dim = 2 # 最终输出 2 维概率
    
    # 模拟一句话，包含 5 个词，每个词 10 维
    X_sentence = np.random.randn(T, input_dim)
    
    rnn = RNN(input_dim, hidden_dim, output_dim)
    
    outputs, hidden_states = rnn.forward(X_sentence)
    
    print(f"经历了 {T} 个时间步的运算...")
    print(f"最后一个时刻的输出 y_T 形状: {outputs[-1].shape} (预期为 (2, 1))")
    print(f"最后一个时刻的记忆 h_T 形状: {hidden_states[-1].shape} (预期为 (6, 1))")

