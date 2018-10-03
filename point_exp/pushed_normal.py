import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


from relie.flow import LocalDiffeoTransformedDistribution as LDTD
from relie.lie_distr import SO3ExpTransform, SO3MultiplyTransform
from relie.utils.so3_tools import so3_exp

# noinspection PyCallingNonCallable
class PushedNormalSO3(nn.Module):
    
    """
    A nn.Module that wraps a Normal pushed to SO(3) 
    having as fixed Parameters the distr parameters
    """
    
    def __init__(self):
        
        super().__init__()
        
        self.pre_scale = nn.Parameter(data=torch.tensor([1.,1.,1.], dtype=torch.float64))
        self.loc = nn.Parameter(torch.tensor([0.,0,0], dtype=torch.float64)+1e-5)
        
    def rsample(self, size):
        """
        sample from distribution
        :param size: size of sample, torch.Size()
        :return:
        """
        alg_distr = Normal(torch.tensor([0.,0.,0.], dtype=torch.float64),
                           F.softplus(self.pre_scale))
        p = LDTD(alg_distr, SO3ExpTransform())
        p = LDTD(p, SO3MultiplyTransform(so3_exp(self.loc)))
                           
        return p.rsample(size)
    
    def log_prob(self, x):
        alg_distr = Normal(torch.tensor([0.,0.,0.], dtype=torch.float64),
                           F.softplus(self.pre_scale))
        p = LDTD(alg_distr, SO3ExpTransform())
        p = LDTD(p, SO3MultiplyTransform(so3_exp(self.loc)))
        
        return p.log_prob(x) 
    
    def kl_div(self, x, prior):
        log_q = self.log_prob(x).double()
        log_p = prior.log_prob(x).double()
        
        return log_q.mean() - log_p.mean()
