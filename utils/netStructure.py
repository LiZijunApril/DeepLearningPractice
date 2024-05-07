import math
from typing import Union

import torch
from torch import Tensor, nn
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
    
# %% 序列到序列模型的实现
class Seq2SeqEncoder(Encoder):
    """用于序列到序列学习的RNN Encoder"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs) -> None:
        super().__init__(**kwargs)
        #* 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)
        
    def forward(self, X, *args):
        # The output 'X' shape: ('batch_size', 'num_steps', 'embed_size')
        X = self.embedding(X)
        # In RNN models, the first axis corresponds to time steps
        X = X.permute(1, 0, 2)
        # When state is not mentioned, it defaults to zeros
        output, state = self.rnn(X)
        #* 'output' shape: ('num_steps', 'batch_size', 'num_hiddens')
        #* 'state' shape: ('num_layers', 'batch_size', 'num_hiddens')
        return output, state

#* 掩码softmax
def masked_softmax(X: Tensor, valid_lens: torch.Tensor):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    #* X: 3D张量，valid_lens: 1D or 2D 张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 在最后一个轴上被掩蔽的元素用一个非常大的负值来替换，指数输出为0
        X= sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        
        return torch.softmax(X.reshape(shape), dim=-1)
        # return nn.functional.softmax(X.reshape(shape), dim=-1)

# 缩放点积注意力
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout:float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout)
    
    # Shape of queries: (batch_size, no. of queries, 'd')
    # Shape of key: (batch_size, no. of key_value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of 'valid_lens': (batch_size) or (batch_size, 查询数)
    def forward(self, queries:Tensor, keys:Tensor, values:Tensor, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True是为了交换keys的后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)

        return torch.bmm(self.dropout(self.attention_weights), values)


# 注意力解码器
class AttentionDecoder(Decoder):
    """The base attention-based decoder interface."""
    def __init__(self, *args, **kwargs) -> None:
        super(AttentionDecoder, self).__init__(*args, **kwargs)
        
    @property
    def attent_weights(self):
        raise NotImplementedError

def transpose_qkv(X: Tensor, num_heads):
    """Transposition for parallel computation of multiple attention heads

        Shape of input 'X': (batch_size, no. of queries or key-value pairs, num_hiddens)
        Shape of output 'X': (batch_size, no. of queries or key-value pairs, num_heads, num_hiddens / num_heads)
    """
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)
    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`."""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class MultiHeadAttention(nn.Module):
    """Multi-head attention"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    
    def forward(self, queries, keys, values, valid_lens):
        #* Shape of 'queris', 'keys', or 'values':
        #* ('batch_size', np. of queries of key-value pairs, 'num_hidens')
        #* Shape of 'valid_lens':
        #* ('batch_size',) or ('batch_size', no. of queries)
        #* After transposing, shape of output 'queries', 'keys' or 'values':

        #* (batch_size * num_heads, no. of queries or key-value pairs, num_hiddens / num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        
        if valid_lens is not None:
            # 在轴0，将第一项（标量或向量）复制num_heads次，然后如此复制第二项，依此类推
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        
        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries, `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)
        
        # Shape of 'output_concat': ('batch_size', no. of queries, num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        
        return self.W_o(output_concat)

# %% 位置编码
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的位置编码
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

# %% 基于位置的前馈网络
class PositionWiseFFN(nn.Module):
    """Position-wise Feed-Forward Network"""

    def __init__(self, ffn_num_input: int, ffn_num_hiddens: int, ffn_num_outputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
        
    def forward(self, X: Tensor):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """残差连接，followed by layer normalization"""
    def __init__(self, normalized_shape: int, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
        
    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module):
    """Transformer encoder block."""
    def __init__(self, key_size: int, 
                 query_size: int, 
                 value_size: int, 
                 num_hiddens: int,
                 norm_shape: Union[tuple, int], 
                 ffn_num_input: int, 
                 ffn_num_hiddens: int, 
                 num_heads: int, 
                 dropout: float,
                 use_bias: bool = False, 
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    
    def forward(self, X: Tensor, valid_lens: Tensor) -> Tensor:
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))