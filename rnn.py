# %%
import math
import torch
from torch import nn 
from torch.nn import functional as F
from utils import dataset, gpu
# %%
batch_size, num_steps = 32, 35
train_iter, vocab = dataset.load_data_time_machine(batch_size, num_steps)

F.one_hot(torch.tensor([0, 2]), len(vocab))

# %%
X = torch.arange(10).reshape((2, 5))
# print(F.one_hot(X.T, 28))
# F.one_hot(X.T, 28).shape

# 初始化模型参数
def get_params(vovab_size, num_hiddens, device):
    num_inputs = num_outputs = vovab_size
    
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    
    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    
    # Output layer paramters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    
    # Attach gradients (附加梯度)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)

    return params

# 为了定义一个RNN，我们首先需要一个 init_rnn_state函数来在初始化时返回隐藏状态。
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# 下面的rnn函数定义了如何在一个时间步内计算隐状态和输出
def rnn(inputs, state, params):
    # inputs的形状为（时间步数，批量大小，词表大小）
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of 'X': ('batch_size', 'vocab_size')
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    
    return torch.cat(outputs, dim=0), (H,)

# 创建一个类打包这些函数
class RNNModelScratch(object):
    def __init__(self, vocab_size,  num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
        
    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)
    
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
    
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, 'mps', get_params, init_rnn_state, rnn)
state = net.begin_state(X.shape[0], 'mps')
Y, new_state = net(X.to('mps'), state)
Y.shape, len(new_state), new_state[0].shape
    

# %%