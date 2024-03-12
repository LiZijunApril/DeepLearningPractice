# %%
import torch
from torch import nn 
from utils import dataset, trainnet
# from rnn import RNNModel
from utils import netStructure

batch_size, num_steps = 32, 35
train_iter, vocab = dataset.load_data_time_machine(batch_size, num_steps)

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    
    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01
    
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))
        
    W_xz, W_hz, b_z = three() # Update gate parameters
    W_xr, W_hr, b_r = three() # Reset gate parameters
    W_xh, W_hh, b_h = three() # Candidate hidden state parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # Attach gradient 附加层参数
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 初始化gru
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# 定义gru
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
        
    return torch.cat(outputs, dim=0), (H,)


DEVICE = 'cpu'
vocab_size, num_hiddens, device = len(vocab), 256, DEVICE
num_epochs, lr = 500, 1
# model = RNNModel(vocab_size, num_hiddens, device, get_params, init_gru_state, gru)
# model = netStructure.RNNModelScratch(vocab_size, num_hiddens, device, get_params, init_gru_state, gru)

# trainnet.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

# 简洁实现
# vocab_size, num_inputs, device = len(vocab), vocab_size, DEVICE
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = netStructure.RNNModel(gru_layer, vocab_size)
model = model.to(device)
trainnet.train_ch8(model, train_iter, vocab, lr, num_epochs, device, show_progress=True)