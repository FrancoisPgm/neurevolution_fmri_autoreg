import torch
import torch.nn as nn

from nets.static.base import StaticNetBase

class Net(StaticNetBase):

    def __init__(self, d_input, d_output, recurrent=True):
        super().__init__()
        self.fc = nn.Linear(d_input, d_output)

    def forward(self, x):
        x = self.fc(x)
        return x
