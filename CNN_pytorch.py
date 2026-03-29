import torch
import torch.nn as nn

# 1. 所有的神经网络模型都必须继承自 nn.Module
class SimpleCNN_PyTorch(nn.Module):
    def __init__(self):
        # 必须调用父类的初始化方法
        super(SimpleCNN_PyTorch, self).__init__()
        
        # 2. 定义网络层 (这和我们自己写的类名几乎一模一样！)
        # nn.Conv2d 自动包含了权重 W 和偏置 b 的初始化
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.relu = nn.ReLU()
        
        # nn.MaxPool2d 处理最大池化
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # nn.Flatten 会自动把除了 batch_size 之外的维度拉平
        self.flatten = nn.Flatten()
        
        # nn.Linear 就是我们之前写的全连接层 (Dense)
        # 这里的 1800 就是你上一节亲自推演出来的心血：8通道 * 15高 * 15宽
        self.fc = nn.Linear(in_features=1800, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 3. 像水流一样拼接前向传播
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# ======= 测试与训练演示 =======
if __name__ == "__main__":
    # 模拟一张图片：注意 PyTorch 的输入通常要带上“批次”维度 (Batch Size)
    # 形状为 (Batch_Size, Channels, Height, Width) -> (1, 3, 32, 32)
    dummy_image = torch.randn(1, 3, 32, 32) 
    
    # 实例化模型
    model = SimpleCNN_PyTorch()
    
    # 前向传播预测
    prediction = model(dummy_image) # 在 PyTorch 中，直接调用 model(x) 等同于 model.forward(x)
    print(f"前向传播预测结果: {prediction.item():.4f}")
    
    # ==========================================
    # 见证魔法的时刻：PyTorch 如何做反向传播？
    # ==========================================
    print("\n--- 准备体验 Autograd 魔法 ---")
    
    # 假设真实的标签是 1.0
    target = torch.tensor([[1.0]]) 
    
    # 定义损失函数：BCE Loss (二元交叉熵)
    criterion = nn.BCELoss()
    
    # 定义优化器：使用随机梯度下降 (SGD)，接管模型的所有参数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 第 1 步：清空上一轮的旧梯度 (极其重要！)
    optimizer.zero_grad()
    
    # 第 2 步：计算损失
    loss = criterion(prediction, target)
    print(f"当前损失 Loss: {loss.item():.4f}")
    
    # 第 3 步：反向传播魔法！(这一行代码替代了你之前手写的所有 backward 函数！)
    loss.backward()
    print("反向传播完成！所有参数的梯度已自动计算。")
    
    # 第 4 步：更新参数 (这一行代码替代了之前的 W -= lr * dW)
    optimizer.step()
    print("参数更新完毕！")