import numpy as np
import torch
import torch.nn as nn


class ConstHypernet(nn.Module):
    def __init__(self, out_size=None, gamma_shift=0, init_gammas=None):
        super().__init__()

        if init_gammas is None:
          self.gammas = np.random.randn(size=out_size) + gamma_shift
        else:
          self.gammas = init_gammas
        
        self.gammas = torch.nn.Parameter(self.gammas, requires_grad=True)
        
    def forward(self, lambd: torch.Tensor):
        return self.gammas + 0
