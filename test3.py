# 写一个简单的线性回归模型，并用pytorch实现。

import torch
import torch.nn as nn
import torch.optim as optim  # 优化器


# 定义模型
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()          # 继承nn.Module
        self.linear = nn.Linear(input_size, output_size)   # 线性层

    def forward(self, x):                                  # 前向传播                
        out = self.linear(x)                               # 线性层输出        
        return out                                       # 输出

# 定义数据集    
x_train = torch.FloatTensor([[1], [2], [3]])            # 输入                    
y_train = torch.FloatTensor([[2], [4], [6]])            # 输出

# 实例化模型        
model = LinearRegression(1, 1)                         # 输入维度1，输出维度1

# 定义��失函数和优化器                                
criterion = nn.MSELoss()                               # 均方误差
optimizer = optim.SGD(model.parameters(), lr=0.01)     # 随机梯度下降优化器

# 训练模型
for epoch in range(1000):                             # 训练1000次
    optimizer.zero_grad()                              # 梯度清零
    outputs = model(x_train)                           # 前向传播
    loss = criterion(outputs, y_train)                  # 计算损失
    loss.backward()                                    # 反向传播                                
    optimizer.step()                                   # 更新参数                                
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 1000, loss.item()))                            # 打印损失

# 预测输出                                
predicted = model(torch.FloatTensor([[4]]))            # 输入为4，预测输出
print('Predicted: {:.4f}'.format(predicted.item()))   # 打印预测输出    