# %% 
import math

import torch
from torch import Tensor, nn, unsqueeze

from utils import netStructure, plot


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
        self.attention = netStructure.DotProductAttention(dropout)
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
    
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
attention.eval()

batch_size, num_queries, num_kvpairs, valid_lens = 2, 4, 6, torch.tensor([3, 2])
X = torch.randn(batch_size, num_queries, num_hiddens)
Y = torch.randn(batch_size, num_kvpairs, num_hiddens)
attention(X, Y, Y, valid_lens).shape

# %% 自注意力
# 写一个自注意力模块，它可以处理输入序列的每个元素与其之前的元素之间的关系。
# 自注意力模块的输入包括输入序列X和输入序列的有效长度valid_lens。
# 输出是序列X的自注意力表示。
# 自注意力模块的计算公式如下：
# 
# $$
# \begin{aligned}
#     \text{Attention}(Q, K, V) &= \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V \\
#     \text{where} \quad Q &= \text{W_q}X, K = \text{W_k}X, V = \text{W_v}X \\
#     \text{and} \quad d_k &= \text{dimension of K}
# \end{aligned}
# $$
# 
# 其中，$X$是输入序列，$Q, K, V$分别是查询、键和值。
# 
# 自注意力模块的输出是经过softmax归一化的$QK^T$与$V$的乘积。
# 
# 自注意力模块的输入、输出和参数都可以是序列，也可以是向量。
# 
# 自注意力模块的实现可以参考[“Attention Is All You Need”](https://arxiv.org/abs/1706.03762)这篇论文。     

num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
attention.eval()
# %%
batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.randn(batch_size, num_queries, num_hiddens)
attention(X, X, X, valid_lens).shape
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
    
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
plot.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
          figsize=(6, 3), legend=["Col % d" %d for d in torch.arange(6, 10)])
# %%
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
plot.show_heatmaps(P, xlabel='Column (encoding dimension)',
                   ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
# %%
 