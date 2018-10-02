import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


from relie.flow import LocalDiffeoTransformedDistribution as LDTD
from relie.lie_distr import SO3ExpTransform, SO3Prior


class PushedNormal(nn.Module): 
    
    """
    A nn.Module that wraps a Normal pushed to SO(3) 
    having as fixed Parameters the distr parameters
    """
    
    def __init__(self):
        
        super().__init__()
        
        self.pre_scale = nn.Parameter( data=torch.tensor([1.,1,1], dtype=torch.float64))
        self.loc = nn.Parameter(torch.tensor([0.,0,0], dtype=torch.float64))
        
    def rsample(self, size):
        
        alg_distr = Normal(self.loc.data, F.softplus(self.pre_scale))
        p = LDTD(alg_distr, SO3ExpTransform())
        return p.rsample(size)
    
    def log_prob(self, x):
        alg_distr = Normal(self.loc.data, F.softplus(self.pre_scale))
        p = LDTD(alg_distr, SO3ExpTransform())
        return p.log_prob(x) 
    
    def kl_div(self, x, prior):
        log_q = self.log_prob(x).double()
        log_p = prior.log_prob(x).double()
        return -log_q.mean() + log_p.mean()
