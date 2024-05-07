from typing import Callable, Union

import torch
import torch.nn as nn
from dataset import load_data_wiki
from matplotlib import pyplot as plt
from netStructure import EncoderBlock
from plot import Accumulator, Animator, Animator_vscode
from timer import Timer
from torch import Tensor


class BERTEncoder(nn.Module):
    """BERT encoder"""
    def __init__(self, 
                 vocab_size: int, 
                 num_hiddens: int, 
                 norm_shape: int, 
                 ffn_num_input: int, 
                 ffn_num_hiddens: int, 
                 num_heads: int, 
                 num_layers: int, 
                 dropout, 
                 max_len=1000,
                 key_size=768,
                 query_size=768,
                 value_size=768,
                 **kwargs):
        super().__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"block{i}", EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入矩阵
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))
    
    def forward(self, tokens, segments, valid_lens):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X   
    
# 掩码语言模型
class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))
        
    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        
        return mlm_Y_hat
    
    
# 下一句预测 (next sentence prediction)
class NextSentencePred(nn.Module):
    """BERT的下一句预测模块"""
    def __init__(self, num_inputs, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X的形状是 (batch_size, 'num_hiddens')
        return self.output(X)
    
# 整合模型

class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, 
                 vocab_size: int,
                 num_hiddens: int,
                 norm_shape: Union[int, tuple],
                 ffn_num_input: int,
                 ffn_num_hiddens: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float,
                 max_len: int = 1000,
                 key_size: int = 768,
                 query_size: int = 768,
                 value_size: int = 768,
                 hid_in_features: int = 768,
                 mlm_in_features: int = 768,
                 nsp_in_features: int = 768,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = BERTEncoder(vocab_size,
                                   num_hiddens,
                                   norm_shape,
                                   ffn_num_input,
                                   ffn_num_hiddens,
                                   num_heads,
                                   num_layers,
                                   dropout,
                                   max_len=max_len,
                                   key_size=key_size,
                                   query_size=query_size,
                                   value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)
        
    def forward(self, tokens: Tensor, 
                segments: Tensor,
                valid_lens: int=None,
                pred_positions: Tensor=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        
        return encoded_X, mlm_Y_hat, nsp_Y_hat
    

# 定义辅助函数，给定训练样本，该函数计算掩码语言模型和下一句预测任务的损失。
# BERT预训练的最终损失是掩码语言模型损失和下一句预测任务损失的和
def _get_batch_loss_bert(net: BERTModel, loss: Callable, vocab_size: int, tokens_X: Tensor, 
                         segments_X: Tensor, valid_lens_x: Tensor, pred_positions_X: Tensor, 
                         mlm_weights_X: Tensor, mlm_Y: Tensor, nsp_y: Tensor):
    
    # 前向传播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X, valid_lens_x.reshape(-1), pred_positions_X)
    
    # 计算掩码语言模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # 预测下一句任务损失
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    
    return mlm_l, nsp_l, l

# 训练BERT模型
def train_bert(net, train_iter, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, Timer()
    animator = Animator_vscode(xlabel='step', ylabel='loss', xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # 掩码语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y = mlm_Y.to(devices[0])
            nsp_y = nsp_y.to(devices[0])
            
            # 计算损失
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X, valid_lens_x, 
                                                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            
            # 记录并打印训练进度
            animator.add(step+1, (metric[0]/metric[3], metric[1]/metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break
            
        print(f'MLM loss {metric[0]/metric[3]:.3f},' 
            f'NSP loss {metric[1]/metric[3]:.3f}')
        print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on {str(devices)}')
    


if __name__ == '__main__':
    # 训练BERT模型
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)
    devices = ['mps']
    loss = nn.CrossEntropyLoss()
    num_steps = 50
    vocab_size = len(vocab)

    net = BERTModel(len(vocab), num_hiddens=128, norm_shape=[128], ffn_num_input=128, 
                    ffn_num_hiddens=256, num_heads=2, num_layers=2, dropout=0.2, key_size=128,
                    query_size=128, value_size=128, hid_in_features=128, mlm_in_features=128,
                    nsp_in_features=128)

    train_bert(net, train_iter, vocab_size, devices, num_steps)
    plt.show()
