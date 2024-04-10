# 长短期记忆 LSTM 从零开始实现
import torch
from torch import nn 
from utils import dataset, netStructure, trainnet

batch_size, num_steps = 32, 35
train_iter, vocab = dataset.load_data_time_machine(batch_size, num_steps)

# 初始化模型参数
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    
    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01
    
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))
        
    W_xi, W_hi, b_i = three() # Input gate parameters
    W_xf, W_hf, b_f = three() # Forget gate parameters
    W_xo, W_ho, b_o = three() # Output gate parameters
    W_xc, W_hc, b_c = three() # Candidate memory cell parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 初始化参数
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))
    
# 实际模型的定义：3个门和一个额外的记忆元，只有隐状态才会传递到输出层，记忆元Ct不直接参与输出计算
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i) # 输入门
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f) # 遗忘门
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o) # 输出门
        # C_tilda = torch.sigmoid((X @ W_xc) + (H @ W_hc) + b_c) # 候选记忆元
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c) # 候选记忆元
        C = F * C + I * C_tilda # 记忆元/元素乘法 
        H = O * torch.tanh(C) # 隐状态
        Y = (H @ W_hq) + b_q # 输出层
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)

vocab_size, num_hiddens, device = len(vocab), 256, 'cpu'
num_epochs, lr = 500, 1
model = netStructure.RNNModelScratch(vocab_size, num_hiddens, device, get_lstm_params, init_lstm_state, lstm)
# trainnet.train_ch8(model, train_iter, vocab, lr, num_epochs, device, show_progress=True)

# 简洁实现
# from utils import d2l
# num_inputs = vocab_size
# lstm_layer = nn.LSTM(num_inputs, num_hiddens)
# model = d2l.RNNModel(lstm_layer, len(vocab))
# model = model.to(device)
# d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)


num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = netStructure.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
trainnet.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

