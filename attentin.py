# %%
import math
from json import decoder
from tkinter import SE

import torch
from sympy import im
from torch import Tensor, nn
from zmq import device

from utils import d2l, dataset, plot, predictor
from utils.gpu import try_gpu
from utils.netStructure import EncoderDecoder, Seq2SeqEncoder, sequence_mask
# from utils.d2l import train_seq2seq
from utils.trainnet import train_seq2seq


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
#* Bahdanau Attention
import torch
from torch import nn

from utils.netStructure import Decoder


class AttentionDecoder(Decoder):
    """The base attention-based decoder interface."""
    def __init__(self, *args, **kwargs) -> None:
        super(AttentionDecoder, self).__init__(*args, **kwargs)
        
    @property
    def attent_weights(self):
        raise NotImplementedError
    
class Seq2SeqAttentionDecoder(AttentionDecoder):
    """The sequence-to-sequence attention decoder."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        #* Shape of 'outputs': ("num_steps", "batch_size", "num_hiddens")
        #* Shape of 'hidden_state[0]': ("num_layers", "batch_size", "num_hiddens")
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        #* Shape of 'enc_outputs': ('batch_size', 'num_steps', 'num_hiddens')
        #* Shape of 'hidden_state[0]': ('num_layers', 'batch_size', 'num_hiddens')
        enc_outputs, hidden_state, enc_valid_lens = state
        #* Shape of the outpus 'X': ('num_steps', 'batch_size', 'emded_size')
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            #* Shape of 'query': ('batch_size', 1, 'num_hiddens')
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            #* Shape of 'context': ('batch_size', 1, 'num_hiddens')
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            #* Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            #* Reshape 'x' as (1, 'batch_size', 'embed_size + num_hiddens')
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention_weights)
        #* After fully-connected layer, transformation, shape of 'outputs': ('num_steps', 'batch_size', 'vocab_size')
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]
    
    @property
    def attention_weights(self):
        return self._attention_weights

# %%
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
decoder.eval()
X = torch.zeros((4, 7), dtype=torch.long) #* (batch_size, num_steps)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
# %%
# from utils import d2l

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
# lr, num_epochs, device = 0.005, 250, try_gpu()
lr, num_epochs, device = 0.005, 250, 'cpu'
train_iter, src_vocab, tgt_vocab = dataset.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = EncoderDecoder(encoder, decoder)

train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
# d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
# %%                                                                        
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = predictor.predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng}==>{translation}, ', 
          f'bleu {predictor.bleu(translation, fra, k=2):.3f}')    
    
# %%
