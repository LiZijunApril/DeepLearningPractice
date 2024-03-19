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
    

class Encoder(nn.Module):
    """基本的编码器接口"""
    def __init__(self, *args, **kwargs) -> None:
        super(Encoder, self).__init__(*args, **kwargs)
        
    def forward(self, X, *args):
        raise NotImplementedError
    
class Decoder(nn.Module):
    """基本的解码器接口"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError
    
    def forward(self, X, state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    """编码器解码器的基本类"""
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
    
# %% 特定的填充词元被添加到序列的末尾，因此不同长度的序列可以以相同的形状的小批量加载。
# 但是我们应该将填充词元的预测在损失函数的计算中剔除
# 使用下面的函数通过零值化屏蔽不相关的项，以便后面任何不相关的预测的计算都是与0的乘积，结果都等于零
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    
    X[~mask] = value #* ‘～’按位取反运算符
    #*原始整数张量: tensor([0, 1, 2, 3, 4])
    #*按位取反后的整数张量: tensor([-1, -2, -3, -4, -5])

    # 根据有效长度 valid_len 创建一个二维的布尔掩码张量。首先，使用 torch.arange 创建一个从 0 到 maxlen-1 的一维张量，然后将其变形为形状为 (1, maxlen) 的二维张量。
    # 接着，使用广播和比较操作，将这个二维张量与 valid_len[:, None] 的二维列向量进行比较，生成一个形状为 (batch_size, maxlen) 的布尔掩码张量，其中 True 表示有效位置，False 表示无效位置。
    # X[~mask] = value：将输入张量 X 中对应掩码为 False 的位置（即无效位置）的元素设置为指定的值 value。
    return X

# %% 扩展softmax交叉熵损失函数来屏蔽不相关的预测。
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带屏蔽的softmax交叉熵损失函数"""
    # pred的形状为(batch_size, num_steps, vocab_size)
    # label的形状为(batch_size, num_steps)
    # valid_len的形状为(batch_size, )
    def forward(self, pred: torch.Tensor, label: torch.Tensor, valid_len: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none' #* self.reduction用于指定损失的计算方式。具体来说有3种取值：
                                #* 'mean': 表示计算损失的平均值
                                #* 'sum': 表示计算损失的综合
                                #* 'none' 表示不进行任何处理，直接返回每个样本的损失值
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        
        return weighted_loss
    
