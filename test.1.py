import torch
# import torch.nn
# from torch import nn

linear = torch.nn.Linear(in_features=64, out_features=1)
input = torch.rand(10, 64)
output = linear(input)
print(output.shape)  # torch.Size([10, 1])

conv1 = torch.nn.Conv1d(in_channels=256, out_channels=10, kernel_size=2, stride=1, padding=0)
input = torch.randn(32, 32, 256)  # [batch_size, L_in, in_channels]
input = input.permute(0, 2, 1)  # 交换维度：[batch_size, embedding_dim, max_len]
out = conv1(input)  # [batch_size, out_channels, L_out]
print(out.shape)  # torch.Size([32, 10, 31]),31=(32+2*0-1*1-1)/1+1

x = torch.randn(3, 1, 5, 4)  # [N, in_channels, H_in, W_in]
conv = torch.nn.Conv2d(1, 4, (2, 3))  # [in_channels, out_channels, kernel_size]
output = conv(x)
print(output.shape)  # torch.Size([3, 4, 4, 2]), [N, out_channels, H_out, W_out]

Bat = torch.nn.BatchNorm1d(2)
input = torch.randn(2, 2)
output = Bat(input)
print(input, output)
# tensor([[ 0.5476, -1.9766],
#         [ 0.7412, -0.0297]]) tensor([[-0.9995, -1.0000],
#         [ 0.9995,  1.0000]], grad_fn=<NativeBatchNormBackward>)

Bat = torch.nn.BatchNorm2d(2)
input = torch.randn(1, 2, 2, 2)
output = Bat(input)
print(input, output)
# tensor([[[[ 0.6798,  0.8453],
#           [-0.1841, -1.3340]],
# 
#          [[ 1.9479,  1.2375],
#           [ 1.0671,  0.9406]]]]) tensor([[[[ 0.7842,  0.9757],
#           [-0.2150, -1.5449]],
# 
#          [[ 1.6674, -0.1560],
#           [-0.5933, -0.9181]]]], grad_fn=<NativeBatchNormBackward>)

feature_size = 32
num_steps = 35
batch_size = 2
num_hiddens = 2
X = torch.rand(num_steps, batch_size, feature_size)
RNN_layer = torch.nn.RNN(input_size=feature_size, hidden_size=num_hiddens)
Y, state_new = RNN_layer(X)
print(X.shape, Y.shape, len(state_new), state_new.shape)
# torch.Size([35, 2, 32]) torch.Size([35, 2, 2]) 1 torch.Size([1, 2, 2])

# 构建4层的LSTM,输入的每个词用10维向量表示,隐藏单元和记忆单元的尺寸是20
lstm = torch.nn.LSTM(input_size=10, hidden_size=20, num_layers=4)
 
# 输入的x:其中batch_size是3表示有三句话,seq_len=5表示每句话5个单词,feature_len=10表示每个单词表示为长10的向量
x = torch.randn(5, 3, 10)
# 前向计算过程,这里不传入h_0和C_0则会默认初始化
out, (h, c) = lstm(x)
print(out.shape)  # torch.Size([5, 3, 20]) 最后一层10个时刻的输出
print(h.shape)  # torch.Size([4, 3, 20]) 隐藏单元
print(c.shape)  # torch.Size([4, 3, 20]) 记忆单元

dconv1 = torch.nn.ConvTranspose1d(1, 1, kernel_size=3, stride=3, padding=1, output_padding=1)
 
x = torch.randn(16, 1, 8)
print(x.size())  # torch.Size([16, 1, 8])
 
output = dconv1(x)
print(output.shape)  # torch.Size([16, 1, 23])

dconv2 = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=3, padding=1, output_padding=1)
 
x = torch.randn(16, 1, 8, 8)
print(x.size()) # torch.Size([16, 1, 8, 8])
 
output = dconv2(x)
print(output.shape) # torch.Size([16, 1, 23, 23])

import torch
# from torch import nn
class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__() # 使用父类的方法初始化子类
        self.linear1 = torch.nn.Linear(96, 1024)  # [96,1024]
        self.relu1 = torch.nn.ReLU(True)
        self.batchnorm1d_1 = torch.nn.BatchNorm1d(1024)
        self.linear2 = torch.nn.Linear(1024, 7 * 7 * 128)  # [1024,6272]
        self.relu2 = torch.nn.ReLU(True)
        self.batchnorm1d_2 = torch.nn.BatchNorm1d(7 * 7 * 128)
        self.ConvTranspose2d = torch.nn.ConvTranspose2d(128, 64, 4, 2, padding=1)
 
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.batchnorm1d_1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.batchnorm1d_2(x)
        x = self.ConvTranspose2d(x)
        return x
 
model = MyNet()
print(model)
# 运行结果为：
# MyNet(
#   (linear1): Linear(in_features=96, out_features=1024, bias=True)
#   (relu1): ReLU(inplace=True)
#   (batchnorm1d_1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (linear2): Linear(in_features=1024, out_features=6272, bias=True)
#   (relu2): ReLU(inplace=True)
#   (batchnorm1d_2): BatchNorm1d(6272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (ConvTranspose2d): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
# )

model = MyNet()  # 实例化的过程中没有传入参数
input = torch.rand([32, 96])  # 输入的最后一个维度要与nn.Linear(96, 1024)中第一个维度96相同
target = model(input)
print(target.shape)  # torch.Size([32, 1, 28, 28])

