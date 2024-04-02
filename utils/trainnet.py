# from utils import timer, plot, evaluation, updater
import math

import evaluation
import plot
import timer
import torch
import trainnet
import updater
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm


def train_ch6(net1, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net1.apply(init_weights)
    print('training on', device)
    net = net1.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = plot.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    timer1, num_batches = timer.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = plot.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer1.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], evaluation.accuracy(y_hat, y), X.shape[0])
            timer1.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluation.evaluate_accuracy_mps(net, test_iter, device)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer1.sum():.1f} examples/sec'
          f'on {str(device)}')
    animator.show(epoch, (None, None, test_acc))
    
def train_per2013(net1, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net1.apply(init_weights)
    print('training on', device)
    net = net1.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = plot.Animator_vscode(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    timer1, num_batches = timer.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = plot.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer1.start()
            optimizer.zero_grad()
            # bs, ncrops, c, h, w = X.size()
            X, y = X.to(device), y.to(device)         
            # print('here2')
            # X = X.view(-1, c, h, w)
            # print(X.shape)
            # break
            y_hat = net(X)
            # print("here")
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], evaluation.accuracy(y_hat, y), X.shape[0])
            timer1.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluation.evaluate_accuracy_mps(net, test_iter, device)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer1.sum():.1f} examples/sec'
          f'on {str(device)}')
    animator.show(epoch, (None, None, test_acc))
    
from utils.predictor import predict_ch8


# 梯度剪裁
def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
              
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a net within one epoch (defined in Chapter 8)."""
    state, timer_lzj = None, timer.Timer()
    metric = plot.Accumulator(2) # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或者使用随机采样时舒适化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
             # state 对于'nn.GRU' 是一个张量
                state.detach_() # 返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个Variable永远不需要计算其梯度，不具有grad。
                            # 即使之后重新将它的requires_grad置为true,它也不会具有梯度grad。
                            # 这样我们就会继续使用这个新的Variable进行计算，后面当我们进行反向传播时，到该调用detach()的Variable就会停止，不能再继续向前进行传播。
            else:
                # state 对于 nn.LSTM 或我们的模型是一个由张量组成的元组
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # Since the 'mean' function has been invoked
            updater(batch_size=1)
        # print('here')
        # print(f'l: {l}, y: {y}, y.numel(): {y.numel()}')
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer_lzj.stop()
# %% 使用高级Api实现的RNN
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False, show_progress=False):
    """训练模型"""
    loss = nn.CrossEntropyLoss()
    animator = plot.Animator_vscode(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, nn.Module):
        updater0 = torch.optim.SGD(net.parameters(), lr)
    else:
        updater0 = lambda batch_size: updater.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    if show_progress == True:
        for epoch in tqdm(range(num_epochs)):
            ppl, speed = train_epoch_ch8(net, train_iter, loss, updater0, device, use_random_iter)
            if (epoch + 1) % 10 == 0:
                print(predict('time traveller'))
                animator.add(epoch + 1, [ppl])
                # animator.show(epoch + 1, [ppl])

    if show_progress == False:
        for epoch in range(num_epochs):
            ppl, speed = train_epoch_ch8(net, train_iter, loss, updater0, device, use_random_iter)
            if (epoch + 1) % 10 == 0:
                print(predict('time traveller'))
                animator.add(epoch + 1, [ppl])
                # animator.show(epoch + 1, [ppl])

    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
    plt.show()  # 显示图形窗口并阻塞程序

# # %% 特定的填充词元被添加到序列的末尾，因此不同长度的序列可以以相同的形状的小批量加载。
# # 但是我们应该将填充词元的预测在损失函数的计算中剔除
# # 使用下面的函数通过零值化屏蔽不相关的项，以便后面任何不相关的预测的计算都是与0的乘积，结果都等于零
# def sequence_mask(X, valid_len, value=0):
#     """在序列中屏蔽不相关的项"""
#     maxlen = X.size(1)
#     mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    
#     X[~mask] = value #* ‘～’按位取反运算符
#     #*原始整数张量: tensor([0, 1, 2, 3, 4])
#     #*按位取反后的整数张量: tensor([-1, -2, -3, -4, -5])
    # return X
    
def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.
    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
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
    # try:
    #     net.to(device)
    # except:
    #     net.to(device)

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