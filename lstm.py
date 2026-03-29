import numpy as np

class LSTM:
    def __init__(self, input_dim, hidden_dim):
        self.d_in = input_dim
        self.d_h = hidden_dim
        self.concat_dim = self.d_h + self.d_in
        
        # 1. 初始化 4 个门的权重矩阵和偏置 (形状均为 d_h x concat_dim 和 d_h x 1)
        self.W_f = np.random.randn(self.d_h, self.concat_dim) * 0.01
        self.b_f = np.zeros((self.d_h, 1))
        
        self.W_i = np.random.randn(self.d_h, self.concat_dim) * 0.01
        self.b_i = np.zeros((self.d_h, 1))
        
        self.W_C = np.random.randn(self.d_h, self.concat_dim) * 0.01
        self.b_C = np.zeros((self.d_h, 1))
        
        self.W_o = np.random.randn(self.d_h, self.concat_dim) * 0.01
        self.b_o = np.zeros((self.d_h, 1))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X_seq):
        T, d_in = X_seq.shape
        
        # 2. 初始化时刻 t=0 的短期记忆 h 和 主传送带 C
        h_prev = np.zeros((self.d_h, 1))
        C_prev = np.zeros((self.d_h, 1))
        
        hidden_states = []
        
        for t in range(T):
            x_t = X_seq[t].reshape(-1, 1)
            
            # TODO 1: 拼接 h_prev 和 x_t (提示：垂直方向堆叠，注意先后顺序)
            concat = np.vstack((h_prev, x_t)) 
            
            # TODO 2: 计算遗忘门 f_t 和 输入门 i_t
            f_t = self._sigmoid(self.W_f @ concat + self.b_f)
            i_t = self._sigmoid(self.W_i @ concat + self.b_i)
            
            # TODO 3: 计算候选记忆 C_tilde
            C_tilde = np.tanh(self.W_C @ concat + self.b_C)
            
            # TODO 4: 更新主传送带 C_t
            C_t = f_t * C_prev + i_t * C_tilde
            
            # TODO 5: 计算输出门 o_t 和 最新的短期记忆 h_t
            o_t = self._sigmoid(self.W_o @ concat + self.b_o)
            h_t = o_t * np.tanh(C_t)
            
            hidden_states.append(h_t)
            
            h_prev = h_t
            C_prev = C_t
            
        return hidden_states

# ======= 测试代码 =======
if __name__ == "__main__":
    np.random.seed(42)
    T, input_dim, hidden_dim = 5, 10, 6
    X_sentence = np.random.randn(T, input_dim)
    lstm = LSTM(input_dim, hidden_dim)
    hidden_states = lstm.forward(X_sentence)
    print(f"最后一个时刻的短期记忆 h_T 形状: {hidden_states[-1].shape} (预期为 (6, 1))")