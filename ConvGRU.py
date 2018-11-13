import torch   
import torch.nn as nn
from torch.autograd import Variable

class ConvGRUCell(nn.Module):
    def __init__(self, input_shape, hidden_c, kernel_shape, batch_size, isGPU):
        """
        input_shape: (channel, h, w)
        hidden_c: the number of hidden channel.
        kernel_shape: (h, w)
        batch_size: batch_size
        isGPU: True/False
        """
        super().__init__()
        self.input_c, self.input_h, self.input_w = input_shape
        self.hidden_c = hidden_c
        self.kernel_h, self.kernel_w = kernel_shape
        self.padding_same_h = self.kernel_h // 2
        self.padding_same_w = self.kernel_w // 2
        self.isGPU = isGPU

        # Initial states for GRU
        self.init_hidden = nn.Parameter(torch.zeros(batch_size, self.hidden_c, self.input_h, self.input_w))
        
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
        if self.isGPU:
            self.init_hidden = Variable(torch.zeros(batch_size, self.hidden_c, self.input_h, self.input_w)).cuda()
        else:
            self.init_hidden = Variable(torch.zeros(batch_size, self.hidden_c, self.input_h, self.input_w))
            
        return self.init_hidden
