import numpy as np
import torch

from bots.static.base import StaticBotBase
from nets.static.fmri_autoreg import Net


class fMRI_Bot(StaticBotBase):

    def __init__(self, d_input, d_output, args, rank):
        self.d_input = d_input
        self.d_output = d_output
        super().__init__(args, rank, pop_nb=1, nb_pops=1)

    def initialize_nets(self):
        self.nets = [Net(self.d_input, self.d_output)]

    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.nets[0](x)
        return x
