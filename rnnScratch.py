# %%
import math
import torch
from utils import d2l
from torch import nn 
from torch.nn import functional as F
from utils import dataset, gpu, timer, plot, updater

DEVICE = 'mps'
# %%
batch_size, num_steps = 32, 35
train_iter, vocab = dataset.load_data_time_machine(batch_size, num_steps)

F.one_hot(torch.tensor([0, 2]), len(vocab))

# %%
X = torch.arange(10).reshape((2, 5))
# print(F.one_hot(X.T, 28))
# F.one_hot(X.T, 28).shape

# 初始化模型参数
def get_params(vovab_size, num_hiddens, device):
    num_inputs = num_outputs = vovab_size
    
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    
    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    
    # Output layer paramters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    
    # Attach gradients (附加梯度)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)

    return params

# 为了定义一个RNN，我们首先需要一个 init_rnn_state函数来在初始化时返回隐藏状态。
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# 下面的rnn函数定义了如何在一个时间步内计算隐状态和输出
def rnn(inputs, state, params):
    # inputs的形状为（时间步数，批量大小，词表大小）
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of 'X': ('batch_size', 'vocab_size')
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    
    return torch.cat(outputs, dim=0), (H,)

# 创建一个类打包这些函数
class RNNModelScratch(object):
    def __init__(self, vocab_size,  num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
        
    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)
    
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
    
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, DEVICE, get_params, init_rnn_state, rnn)
state = net.begin_state(X.shape[0], DEVICE)
Y, new_state = net(X.to(DEVICE), state)
Y.shape, len(new_state), new_state[0].shape
    
# %% 预测 在循环遍历prefix中的初始字符时，我们不断地将隐状态传递到下一个时间步，但是不生成任何输出。这被称为预热，
# 在此期间模型会进行更新，但不会进行预测。预热期结束后，隐状态的值通常比初始值更是预测
def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]: # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds): # Predict 'num_preds' steps
        y, state = net(get_input(), state)
        # outputs.append(int(y.argmax(dim=1).reshape(1)))
        _, y = y.max(dim=1)
        outputs.append(int(y.reshape(1)))

    return ''.join([vocab.idx_to_token[i] for i in outputs])

predict_ch8('time traveller ', 10, net, vocab, DEVICE)

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
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
      
num_epochs, lr = 500, 1
# train_ch8(net, train_iter, vocab, lr, num_epochs, DEVICE)
net = RNNModelScratch(len(vocab), num_hiddens, DEVICE, get_params, init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, DEVICE, use_random_iter=True)
# %%
