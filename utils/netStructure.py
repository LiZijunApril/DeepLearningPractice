import torch
from torch import nn 
from torch.nn import functional as F

# RNN 简洁实现
class RNNModel(nn.Module):
    """The RNN model"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的，num_direction应该是2，否则是1
        if not self.rnn.bidirectional:
            self.num_direactions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_direactions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)
            
    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改成（时间步数*批量大小，隐藏单元数）
        # 它的输出形状为（时间步数*批量大小， 词表大小）
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state
    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU 以张量作为隐藏层
            return torch.zeros((self.num_direactions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        else:
            # # nn.LSTM 以元组作为隐藏层
            return (torch.zeros((self.num_direactions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device), torch.zeros((self.num_direactions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device))

# RNN 从零开始实现
class RNNModelScratch(object):
    def __init__(self, vocab_size,  num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
        
    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)
    
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)# 下面的rnn函数定义了如何在一个时间步内计算隐状态和输出