import torch   
import torch.nn as nn
from torch.autograd import Variable
import sys
sys.path.append('/export/home/choi574/git_libs/misc/')
import misc

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
                out_channels=2, #self.hidden_c * 2, ## Changed for less GATE
                kernel_size=(self.kernel_h, self.kernel_w), 
                stride=1, 
                padding=(self.padding_same_h, self.padding_same_w),
                padding_mode=pad_mod)
        self.norm = nn.GroupNorm(GN*2, self.hidden_c*2)

        self.gate_channel = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                misc.Flatten(),
                nn.Linear(self.input_c + self.hidden_c, 64), 
                nn.ReLU(), 
                nn.Linear(64, self.hidden_c*2))
                

        self.in_conv = nn.Conv2d(in_channels=self.input_c + self.hidden_c, 
                out_channels=self.hidden_c, 
                kernel_size=(self.kernel_h, self.kernel_w), 
                stride=1, 
                padding=(self.padding_same_h, self.padding_same_w),
                padding_mode=pad_mod) # 20200918 modified. 
        self.norm_in = nn.GroupNorm(GN, self.hidden_c)

    def forward(self, input_cur, state_prev):
        input_concat = torch.cat([input_cur, state_prev], dim=1) # (batch, c, h, w)
        #gates = self.norm(self.gate_conv(input_concat))
        gates = self.gate_conv(input_concat) # (b, 2, h, w)
        gate_update, gate_reset = gates.chunk(2, 1) # (b, 1, h, w)
        gate_update = torch.sigmoid(gate_update)
        gate_reset = torch.sigmoid(gate_reset)

        gates_c = self.gate_channel(input_concat) # (b, 2*hidden_c)
        gate_update_c, gate_reset_c = gates_c.chunk(2, 1) # (b, hidden_c)
        gate_update_c = torch.sigmoid(gate_update_c).unsqueeze(-1).unsqueeze(-1) # (b, hidden_c, 1, 1)
        gate_reset_c = torch.sigmoid(gate_reset_c).unsqueeze(-1).unsqueeze(-1) # (b, hidden_c, 1, 1)


        #input_concat_reset = torch.cat([input_cur, state_prev*gate_reset*gate_reset_c], dim=1)
        input_concat_reset = torch.cat([input_cur, state_prev*gate_reset], dim=1)
        in_state = self.norm_in(self.in_conv(input_concat_reset))
        
        if self.active_fn == 'tanh':
            in_state = torch.tanh(in_state)
        elif self.active_fn == 'relu':
            in_state = self.relu(in_state)
        else:
            print('error ConvGRU activation function not defined')

        state_cur = state_prev * (1 - gate_update) * (1-gate_update_c) + in_state * gate_update * gate_update_c
        #state_cur = state_prev * (1 - gate_update*gate_update_c) + in_state * (gate_update * gate_update_c)

        return state_cur
