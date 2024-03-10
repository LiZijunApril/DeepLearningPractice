# from utils import timer, plot, evaluation, updater
import timer, plot, evaluation, updater
from matplotlib import pyplot as plt
import torch
import math
from torch import nn

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
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
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