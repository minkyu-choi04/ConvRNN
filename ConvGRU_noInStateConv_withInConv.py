import torch   
import torch.nn as nn
from torch.autograd import Variable

class ConvGRUCell(nn.Module):
    def __init__(self, input_shape, hidden_c, kernel_shape, active_fn='tanh', pad_mod='replicate', GN=32):
        """
        input_shape: (channel, h, w)
        hidden_c: the number of hidden channel.
        kernel_shape: (h, w)
        avtive_fn: 'tanh' or 'relu'
        """
        super().__init__()
        self.active_fn = active_fn
        self.relu = nn.ReLU()
        self.input_c, self.input_h, self.input_w = input_shape
        self.hidden_c = hidden_c
        self.kernel_h, self.kernel_w = kernel_shape
        self.padding_same_h = self.kernel_h // 2
        self.padding_same_w = self.kernel_w // 2

        # Initial states for GRU
        self.init_hidden = nn.Parameter(torch.randn(1, self.hidden_c, self.input_h, self.input_w), requires_grad=True)
        
        self.gate_conv = nn.Conv2d(in_channels=self.input_c + self.hidden_c,
                out_channels=self.hidden_c * 2, 
                kernel_size=(self.kernel_h, self.kernel_w), 
                stride=1, 
                padding=(self.padding_same_h, self.padding_same_w),
                padding_mode=pad_mod)
        self.norm = nn.GroupNorm(GN*2, self.hidden_c*2)

        self.in_conv = nn.Conv2d(in_channels=self.input_c-1, 
                out_channels=self.hidden_c, 
                kernel_size=(self.kernel_h, self.kernel_w), 
                stride=1, 
                padding=(self.padding_same_h, self.padding_same_w),
                padding_mode=pad_mod) # 20200918 modified. 
        self.norm_in = nn.GroupNorm(GN, self.hidden_c)

    def forward(self, input_cur, ior, state_prev):
        input_cur_ior = torch.cat((input_cur, ior), 1)
        input_concat = torch.cat([input_cur_ior, state_prev], dim=1) # (batch, c, h, w)
        gates = self.norm(self.gate_conv(input_concat))
        gate_update, gate_reset = gates.chunk(2, 1)
        gate_update = torch.sigmoid(gate_update)
        gate_reset = torch.sigmoid(gate_reset)

        #input_concat_reset = torch.cat([input_cur, state_prev*gate_reset], dim=1)
        in_state = self.norm_in(self.in_conv(input_cur))
        #in_state = input_cur
        
        if self.active_fn == 'tanh':
            in_state = torch.tanh(in_state)
        elif self.active_fn == 'relu':
            in_state = self.relu(in_state)
        else:
            print('error ConvGRU activation function not defined')

        state_cur = state_prev * (1 - gate_update) + in_state * gate_update

        return state_cur
