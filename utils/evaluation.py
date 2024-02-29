from utils import plot
import torch
from torch import nn

size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)

def accuracy(y_hat, y):
    """计算正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # y_hat = y_hat.argmax(axis=1)
        _, y_hat = y_hat.max(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    metric = plot.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def evaluate_accuracy_mps(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval() #设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测数量
    metric = plot.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT 微调
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            # metric.add(accuracy(net(X), y), y.numel())
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]