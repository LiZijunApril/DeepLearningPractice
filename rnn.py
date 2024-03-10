# %%
import torch
from torch import nn 
from torch.nn import functional as F 
from utils import dataset, predictor, trainnet

batch_size = 32
num_steps = 35
train_iter, vocab = dataset.load_data_time_machine(batch_size, num_steps)

# 高级api实现rnn
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)

# 使用张量来初始化状态，它的形状是(1, batch_size, num_hiddens)
state = torch.zeros((1, batch_size, num_hiddens))
state.shape
# %%
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, state_new.shape
# %% 为完整的循环神经网络定义一个RNNModel类。rnn_layer只包含隐藏的循环层。我们还需要创建一个单独的输出层
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
            # nn.LSTM 以元组作为隐藏层
            return (torch.zeros((self.num_hiddens * self.rnn.num_layers, batch_size, self.num_hiddens), device=device), torch.zeros((self.num_direactions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device))
        
DEVICE = 'mps'
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(DEVICE)
predictor.predict_ch8('time traveller', 10, net, vocab, DEVICE)
# %%
num_epochs, lr = 500, 1
trainnet.train_ch8(net, train_iter, vocab, lr, num_epochs, DEVICE)
# %%000
