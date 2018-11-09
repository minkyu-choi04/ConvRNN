import ConvLSTM
import ConvGRU
import web_convlstm
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

# input description
step_size = 10 
batch_size = 1
channel_size = 1 
height = 5
width = 5
# model description
hidden_size = 10
kernel_size = 3
isLSTM = True#: GRU
# learning description
learning_rate = 0.1
isGPU = False 
epoch_size = 1500
# Convolution for out from RNN state
conv_out = nn.Conv2d(in_channels=hidden_size, 
        out_channels=channel_size,
        kernel_size=kernel_size, 
        padding=kernel_size//2)

print('Set model layer\n')
if isLSTM:
    layer1 = ConvLSTM.ConvLSTMCell((channel_size, height, width), hidden_size, (kernel_size, kernel_size), isGPU)
else:
    layer1 = ConvGRU.ConvGRUCell((channel_size, height, width), hidden_size, (kernel_size, kernel_size), isGPU)

print('Set input/target\n')
data = Variable(torch.rand(step_size, batch_size, channel_size, height, width))

print('Set criterion and optimizer\n')
criterion = nn.MSELoss()
optimizer = optim.Adam(list(layer1.parameters())+list(conv_out.parameters()), lr=learning_rate)

print('Select GPU/CPU\n')
if isGPU:
    print('Select GPU\n')
    layer1 = layer1.cuda()
    data = data.cuda()
    conv_out.cuda()
    criterion.cuda()
else:
    print('Select CPU\n')

print('Start training\n')
for epoch in range(epoch_size):
    state_prev = layer1.set_init_hidden_state(batch_size)
    loss = 0
    out_list = []
    for step in range(step_size-1):
        state_cur = layer1(data[step], state_prev)
        pred = conv_out(state_cur[0]) if isLSTM else conv_out(state_cur)
        loss += criterion(pred, data[step+1])
        out_list.append(pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('%d epoch, loss: %.3f\n' % (epoch, loss.item()/height/width))

print('Sample prediction output\n')
print(pred)
print('Sample target\n')
print(data[step+1])
print('End training\n')





