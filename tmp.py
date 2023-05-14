import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear, Conv1d, ReLU


class WavenetEncoder(nn.Module):
    def __init__(self, input_size, kernel_size):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        for i in range(7):
            setattr(self, 'conv1D_%d' % i,
                    Conv1d(1, 1, kernel_size=kernel_size, stride=1, dilation=2 ** i))
        self.relu = ReLU()
        self.linear = Linear()

    def forward(self, x):
        for i in range(7):
            x = getattr(self, 'conv1D_%d' % i)(x)
            x = self.relu(x)
            # output_size = self.input_size - (2 ** i) * (self.kernel_size - 1)
        return x

wavencoder = WavenetEncoder(100, 3)
x = torch.arange(1000).reshape(10, 1, 100).type(torch.float)
output = wavencoder(x)