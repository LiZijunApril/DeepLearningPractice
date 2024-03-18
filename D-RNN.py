# %%
import torch
from torch import nn 
from utils import dataset, netStructure, trainnet

batch_size, num_steps = 32, 35
train_iter, vocab = dataset.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = 'mps'
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = netStructure.RNNModel(lstm_layer, len(vocab))
model = model.to(device)

num_epochs, lr = 500, 2
trainnet.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
# %%
