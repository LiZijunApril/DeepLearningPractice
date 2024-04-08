# %%
import torch
import math
import collections
import logging
from torch import nn
from utils import plot, timer,  trainnet, dataset, gpu, log

class Encoder(nn.Module):
    """基本的编码器接口"""
    def __init__(self, **kwargs) -> None:
        super(Encoder, self).__init__(**kwargs)
        
    def forward(self, X, *args):
        raise NotImplementedError
    
class Decoder(nn.Module):
    """基本的解码器接口"""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
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

# %%
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)
output, state = encoder(X)
output.shape
# %%
class Seq2SeqDecoder(Decoder):
    """序列到序列学习的RNN Decoder"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]
    
    def forward(self, X, state):
        #* The output 'X' shape: ('num_steps', 'batch_size', 'embed_size')
        X = self.embedding(X).permute(1, 0, 2)
        # 广播 'context' 使其具有与'X'相同的时间步
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        #* 'output' shape: ('batch_size', 'num_steps', 'vocab_size')
        #* 'state' shape: ('num_layers', 'batch_size', 'num_hiddens')
        return output,state
    
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
output.shape, state.shape

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

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
sequence_mask(X, torch.tensor([1, 2]))
# %% 
X = torch.ones(2, 3, 4)
X
# %%
sequence_mask(X, torch.tensor([1, 0]), value=0)
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
    
loss = MaskedSoftmaxCELoss()
loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long), torch.tensor([4, 2, 0]))

# %% 在下面的循环训练过程中，特定的序列开始词元<bos>和原始的输出序列（不包括序列结束词元'<eos>'）
# 连接在一起作为解码器的输入，这被称为(teaching force),因为原始的输出序列（词元的标签）被送入解码器，
# 或者将来自上一个时间步的预测得到的词元作为解码器的当前输入
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    try:
        net.to(device)
    except:
        net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss =  MaskedSoftmaxCELoss()
    net.train()
    animator = plot.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    
    for epoch in range(num_epochs):
        timerr = timer.Timer()
        metric = plot.Accumulator(2) # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1) # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward() # 损失函数的标量进行“反向传播”
            trainnet.grad_clipping(net, 1) #? 梯度剪裁？什么意思？
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timerr.stop():.1f} '
          f'tokens/sed on {str(device)}')
    
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 128, 5    
lr, num_epochs, device = 0.005, 300, gpu.try_gpu()
train_iter, src_vocab, tgt_vocab = dataset.load_data_nmt(batch_size, num_steps)


encoder = Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = EncoderDecoder(encoder, decoder)

train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

# %% 预测
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab: dataset.Vocab, num_steps, device, save_attention_weights=False):
    """Predict for sequence to sequence"""
    #* Set 'net' to eval mode for inference
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = dataset.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    #* Add the batch size
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    #* Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用预测可能性最大的词元，作为解码器在下一个时间步的输入
        # dec_X = Y.argmax(dim=2)
        dec_X = Y.argmax(dim=2)

        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦<eos>被预测出，输出序列的预测就结束了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

import collections
import logging
from utils import log

logger = logging.getLogger("machine_translate")
log.setup_logging()
logger = logging.getLogger("RNN_log")
# embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
# batch_size, num_steps = 4, 10
# lr, num_epochs, device = 0.005, 300, gpu.try_gpu()
logger.info(f'embed_size => {embed_size},  num_hiddens => {num_hiddens}, num_layers => {num_layers},  dropout => {dropout},  batch size ==> {batch_size},  num_steps ==> {num_steps}')
# %% BLEU
def bleu(pred_seq: str, label_seq: str, k: str):
    """Compute the BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k+1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1          
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score



engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']

for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device
    )
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
    logger.info(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
# %%
class NWkernelRegiresion(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
        
    def forward(self, queries, keys, values):
        # Shape of the output 'queries' and 'attention_weigths' is (no. of queries, no. of key-value paris)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weigths = nn.functional.softmax(-((queries - keys)*self.w)**2 /2, dim=1)
        # Shape of 'values': (no. of queries, no of key-value pairs)
        return torch.mm(self.attention_weigths.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)
    
n_train = 50 # No. of traning example
x_train, _ = torch.sort(torch.rand(n_train) * 5)

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,)) # Training outputs

# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = x_train.repeat((n_train, 1))
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = y_train.repeat((n_train, 1))
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# %%