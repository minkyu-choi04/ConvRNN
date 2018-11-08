// Tensor and Variable in pytorch are different. Variable is wrapper of tensor. Tensor is just 'data' storing values. And variable is including tensor and having more things like relationships between variables and gradients. Therefore, when calculating gradient to do BP, we need to use variable, not tensor. But when we need just tensor from variable, using .data would output tensor. 
// a = b.data

import torch
import torch.nn as nn
from torch.autograd import Variable

class ConvGPUCell(nn.Module):
    def __init__(self, input_shape, hidden_c, kernel_shape):
        """
        input_shape: (channel, h, w)
        hidden_c: the number of hidden channel.
        kernel_shape: (h, w)
        """
        super().__init__()
        self.input_c, self.input_h, self.input_w = input_shape
        self.hidden_c = hidden_c
        self.kernel_h, self.kernel_w = kernel_shape
        self.padding_same_h = self.kernel_h // 2
        self.padding_same_w = self.kernel_w // 2
        
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) 
        self.gate_conv = nn.Conv2d(in_channels=self.input_c + self.hidden_c,
                out_channels=self.hidden_c * 2, 
                kernel_size=(self.kernel_h, self.kernel_w), 
                stride=1, 
                padding=(self.padding_same_h, self.padding_same_w))
        self.in_conv = nn.Conv2d(in_channels=self.input_c + self.hidden_c, 
                out_channels=self.hidden_c, 
                kernel_size=(self.kernel_h, self.kernel_w), 
                stride=1, 
                padding=(self.padding_same_h, self.padding_same_w))

    def forward(self, input_cur, state_prev):
        input_concat = torch.cat([input_cur, state_prev], dim=1) # (batch, c, h, w)
        gates = self.gate_conv(input_concat)
        gate_update, gate_reset = gates.chunk(2, 1)
        gate_update = torch.sigmoid(gate_update)
        gate_reset = torch.sigmoid(gate_reset)

        input_concat_reset = torch.cat([input_cur, state_prev*gate_reset], dim=1)
        in_state = self.in_conv(input_concat_reset)
        in_state = torch.tanh(in_state)

        state_cur = state_prev * (1 - gate_update) + in_state * gate_update

        return state_cur

    def set_init_hidden_state(self, batch_size):
        self.init_hidden = (Variable(torch.zeros(batch_size, self.hidden_c, self.hidden_h, self.hidden_w)).cuda(), 
            Variable(torch.zeros(batch_size, self.hidden_c, self.hidden_h, self.hidden_w)).cuda())
        return self.init_hidden
