import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/')
import Gaussian2d_mask_generator_v1 as G

class ConvAccumCell(nn.Module):
    def __init__(self, input_c, hidden_c, input_s=(14,14)):
        super().__init__()
        self.input_s = input_s
        self.conv_out = nn.Conv2d(hidden_c, hidden_c, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.conv_in = nn.Conv2d(input_c, hidden_c, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.norm_out = nn.GroupNorm(32, hidden_c)
        self.norm_in = nn.GroupNorm(32, hidden_c)
        self.relu = nn.ReLU()
        self.sigma = 25
        

    def forward(self, fixs, inputs, state_prev=None):
        '''
        inputs
            inputs: (b, c, h, w), tensor
            state_prev: (b, c, h, w), tensor
            fixs: (b, 2), tensor ranged in -1~1, (float x, float y)
        Returns:
            state_curr: (b, c, h, w), tensor
        '''
        gauss_kernel = G.get_gaussian_kernel(fixs, kernel_size=self.input_s, sigma=self.sigma, channels=1, norm='max')
        # (b, 1, 14, 14)

        state_in = self.relu(self.norm_in(self.conv_in(inputs)))

        if state_prev is None: # It means step==0
            blend = state_in
        else:
            blend = (state_in * gauss_kernel) + (state_prev * (1-gauss_kernel))
        state_curr = self.relu(self.norm_out(self.conv_out(blend)))
        return state_curr



