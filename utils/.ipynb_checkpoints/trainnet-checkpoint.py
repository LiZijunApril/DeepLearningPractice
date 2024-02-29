from utils import timer, plot, evaluation
import torch
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