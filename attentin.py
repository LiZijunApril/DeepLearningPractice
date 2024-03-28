import math
import torch
from torch import Tensor, nn

from utils import plot
from utils.netStructure import sequence_mask

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
    
#%%
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
# %%
masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]]))
# %%
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size: int, query_size: int, num_hiddens: int, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries:Tensor , keys:Tensor, values:Tensor, valid_lens:Tensor):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 维度扩展后，‘queries’的形状为（batch_size, 查询数，1，num_hidden）
        # key的形状为（batch_size, 1, 键值对数，num_hiddens）
        # 使用广播方式求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        
        scores = self.w_v(features).squeeze(-1)
        # print(scores.shape, scores)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状为(batch_size, 键值对数，值的降维)
        return torch.bmm(self.dropout(self.attention_weights), values)

# %%
queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# values的小批量，两个值矩阵事相同的
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
valid_lens = torch.tensor([2, 6])
attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
attention.eval()
attention(queries, keys, values, valid_lens)
# %%
plot.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')
# %% 
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
    

queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
attention(queries, keys, values, valid_lens)


# %%
