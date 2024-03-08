#%%
import torch
from torch import nn 
from utils import plot, dataset, evaluation
from loguru import logger
import random

logger.add('./rnnLog.log')

#%%
T = 1000
time1 = torch.arange(1, T+1, dtype=torch.float32)
x = torch.sin(0.01 * time1) + torch.normal(0, 0.2, (T,))
# plot.plot(time1, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
# logger.info('普通消息')

# %%
tau = 4
features = torch.zeros((T - tau, tau))
# print(features.shape)
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))
batch_size, n_train = 16, 600

train_iter = dataset.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)

# %%
# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

loss = nn.MSELoss(reduction='none')
        
# %% 训练模型
def train(net, train_iter, loss, epochs, lr):
    logger.info("Training start")
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        logger.info(f'epoch {epoch+1}, '
                    f'loss: {evaluation.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)

# %% 预测
onestep_preds = net(features)
# plot.plot([time1, time1[tau:]], [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
        #   'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))

# %%
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(multistep_preds[i - tau:i].reshape((1, -1)))
    
# plot.plot([time1, time1[tau:], time1[n_train + tau:]], 
#           [x.detach().numpy(), onestep_preds.detach().numpy(),
#            multistep_preds[n_train + tau:].detach().numpy()], 'time',
#           'x', legend=['data', '1-step preds', 'multistep preds'],
#           xlim=[1, 1000], figsize=(6, 3))
# # 结果不理想，因为误差会积累

# %%
max_steps = 64
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i+1）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列i(i>=tau)是来自（i-tau+1）步的预测，其时间步从（i+1)到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)
    
steps = (1, 4, 16, 64)
plot.plot([time1[tau + i - 1: T - max_steps + i] for i in steps], 
          [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
          legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000], figsize=(6, 3))

# %%
import random
import torch
from utils import dataset, plot

# dataset.DATA_HUB['time_machine'] = ('http://d2l-data.s3-accelerate.amazonaws.com/' + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')
tokens = dataset.tokenize(dataset.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，所以我们吧所有文本行连接在一起
corpus = [token for line in tokens for token in line]
vocab = dataset.Vocab(corpus)
vocab.token_freqs[:10]

freqs = [freq for token, freq in vocab.token_freqs]
# plot.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)', xscale='log', yscale='log')
# plot.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)', xscale='linear', yscale='linear')

# 二元语法
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = dataset.Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]

# 三元语法
trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = dataset.Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]

# 对比3中模型中的词元频率
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
plot.plot([freqs, bigram_freqs, trigram_freqs], xlabel='tokens: x', ylabel='frequency: n(x)', xscale='log',
          yscale='log', legend=['unigram', 'bigram', 'trigram'])
# %%
# 读取长序列数据
# 1.随机抽样
def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中
    # 来自两个相邻的，随机的，小批量的子序列不一定在原始序列中相邻
    random.shuffle(initial_indices)
    
    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]
    
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
        
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY: ', Y)
# %%
# 2. 顺序分区
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始分区
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
        
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY: ', Y)

# %%
# 把上面两个抽样函数包装到一个类中
from utils import dataset
class SeqDataLoader:
    """加载数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens) -> None:
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = dataset.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps
        
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
    
def load_data_time_machine(batch_size, num_steps, use_random_itr=False, max_tokens=1000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_itr, max_tokens)
    return data_iter, data_iter.vocab

load_data_time_machine(128, 5)
# %%
import torch

X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))

# 对X, W_xh进行矩阵乘法，对H, W_hh矩阵乘法，结果相加 
print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))
# 对X，H进行沿列的连接（形成一个(3,5)的矩阵），再对W_xh和W_hh进行沿行的连接（形成一个(5,4)的矩阵），两个矩阵相乘
print(torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0)))
# 结果是一样的
print("等价")
 # %%
