#%%
import torch
from torch import nn 
from utils import plot, dataset, evaluation
from loguru import logger

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
plot.plot([time1, time1[tau:]],
          [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
          'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))