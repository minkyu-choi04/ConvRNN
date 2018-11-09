import torch
import torch.nn as nn
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):
    def __init__(self, input_shape, hidden_c, kernel_shape, isGPU):
        """
        input_shape: (channel, h, w)
        hidden_c: the number of hidden channel.
        kernel_shape: (h, w)
        isGPU: True/False
        """
        super().__init__()
        self.input_c, self.input_h, self.input_w = input_shape
        self.hidden_c = hidden_c
        self.kernel_h, self.kernel_w = kernel_shape
        self.padding_same_h = self.kernel_h // 2
        self.padding_same_w = self.kernel_w // 2
        self.isGPU = isGPU
        
        self.gate_conv = nn.Conv2d(in_channels=self.input_c + self.hidden_c,
                out_channels=self.hidden_c * 4, 
                kernel_size=(self.kernel_h, self.kernel_w), 
                stride=1, 
                padding=(self.padding_same_h, self.padding_same_w))



        self.cal =  nn.Conv2d(in_channels=hidden_c, 
                out_channels=input_shape[0], 
                kernel_size=(3,3), 
                stride=1,
                padding=(3//2, 3//2))

    def forward(self, input_cur, state_prev):
        hidden_prev, cell_prev = state_prev
        input_concat = torch.cat([input_cur, hidden_prev], dim=1) # (batch, c, h, w)
        gates = self.gate_conv(input_concat)
        gate_input, gate_forget, gate_output, gate_cell = gates.chunk(4, 1)

        gate_input = torch.sigmoid(gate_input)
        gate_forget = torch.sigmoid(gate_forget)
        gate_output = torch.sigmoid(gate_output)
        gate_cell = torch.tanh(gate_cell)

        cell_cur = (gate_forget * cell_prev) + (gate_input * gate_cell)
        hidden_cur = torch.tanh(cell_cur) * gate_output

        return hidden_cur, cell_cur

    def set_init_hidden_state(self, batch_size):
        if self.isGPU:
            self.init_hidden = (Variable(torch.zeros(batch_size, self.hidden_c, self.input_h, self.input_w).cuda()), Variable(torch.zeros(batch_size, self.hidden_c, self.input_h, self.input_w)).cuda())
        else:
            self.init_hidden = (Variable(torch.zeros(batch_size, self.hidden_c, self.input_h, self.input_w)), Variable(torch.zeros(batch_size, self.hidden_c, self.input_h, self.input_w)))
        return self.init_hidden

