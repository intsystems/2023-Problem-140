import numpy as np
import torch

from mylib.nas.module2graph import GraphInterperterWithGamma

class GraphInterperterWithGumbelSoftmaxGamma(GraphInterperterWithGamma):
    def __init__(self, mod, gamma_shift=0.0, temperature=1.0):
        self.gamma_shift = gamma_shift
        self.temperature = temperature
        super().__init__(mod)

    def init_gammas(self):
        i = 0
        gammas = []
        self.gammas_name = {}
        for node in self.graph.nodes:
            if node.op == 'call_module':
                gammas.append(np.random.randn()+self.gamma_shift)
                self.gammas_name[str(node)] = i # `str(node)` just for the convenience
                i += 1
        self.gammas = torch.as_tensor(gammas)
        self.discrete = False 

    def sample_gammas(self, previous=False):
        if self.discrete:
            return self.gammas
        if not previous:
            rb = torch.distributions.RelaxedBernoulli(logits=self.gammas, temperature=self.temperature)
            self.last_sample = rb.rsample()
        return self.last_sample
        
    def make_gammas_discrete(self):
        self.gammas = (self.gammas.clone().detach()>=0) * 1.0
        self.gammas.requires_grad = False 
        self.discrete = True
        
    def make_gammas_relaxed(self, gammas=None):
        if gammas is not None:
            self.gammas = gammas
        self.discrete = False
