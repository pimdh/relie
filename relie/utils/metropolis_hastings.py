import torch

from torch.distributions import Normal, Uniform

from relie.flow import LocalDiffeoTransformedDistribution as LDTD
from relie.lie_distr import SO3ExpTransform, SO3Prior, SO3MultiplyTransform

def so3_kernel_gen(centers, scale = torch.ones(3).double()*0.1):
    """
    Given centers it outputs transition kernels around the centers
    the kernels will be normals centered at <centers>
    
    centers: tensor of shape (..., 3, 3) of SO(3) elements 
    scale: tensor of shape (3) of double precision positive floats
    
    returns: a torch.distribution
    """
    
    
    expand_dims = centers.shape[:-2]
    exp_scale = scale.expand(*expand_dims,-1).double()
    
    loc = torch.zeros(3).expand(*expand_dims,-1).double()
    alg_distr = Normal(loc, scale)
    ker = LDTD(alg_distr, SO3ExpTransform(k_max = 20))
    ker = LDTD(ker, SO3MultiplyTransform(centers))
    return ker

def r_kernel_gen(centers, scale = torch.ones(1).double()):
    """
    Given centers it outputs transition kernels around the centers
    the kernels will be normals centered at <centers>
    
    centers: tensor of shape (..., 1) of real numbers
    scale: tensor of shape (1) of double precision positive floats
    
    returns: a torch.distribution
    """
    
    dim = centers.shape[0]
    exp_scale = scale.expand(dim, -1).double()
    ker = Normal(centers, exp_scale)
    return ker

def mh_step(x, log_energy, kernel_gen):
    """
    Given a vectorofstarting points it perform a step of the metropolis hastings algorithm
    to sample log_energy 
    
    x: tensor of shape (batch_dims, distr_dims) of real numbers representing initial samples
    log_energy : function that given a batch (batch_dims, distr_dims) of points 
                    returns a tensor of shape (batch_dims) ofthe log_energy of each point
    kernel_gen: function that taken a batch of points, returns a kernel distr centered at those points
                in form of a torch.distribution
    
    returns: a tensor of shape (batch_dims, distr_dims) of the new samples
    """
    
    s = torch.Size((1,))
    x = x.double()
    ker = kernel_gen(x)
    x1 = ker.sample(s).squeeze(0)
    ker1 = kernel_gen(x1)
    
    log_p_x1_x = ker.log_prob(x1).double()
    log_p_x_x1 = ker1.log_prob(x).double()
    log_acceptance = log_energy(x1) - log_energy(x) + log_p_x_x1 - log_p_x1_x
    u = Uniform(torch.zeros(log_acceptance.shape).double(), torch.ones(log_acceptance.shape).double())
    
    acceptance_mask = u.sample(s).log().squeeze(0) <= log_acceptance
    
    x[acceptance_mask] = x1[acceptance_mask]
    
    return x

def mh(log_energy, lenght, n_chains, kernel_gen, prior, burnin = 0):
    
    """
    Given log_energ, a transition kernel genrator and a prior i runs the metropolis hastings algorithm
    for <lenght> steps and <n_chains> markov chains, discading the first <burnin> samples
      
    log_energy : function that given a batch (batch_dims, distr_dims) of points 
                    returns a tensor of shape (batch_dims) ofthe log_energy of each point                
    lenght: lenght of each markov chain   
    n_chains: number of parallel markov chains that needto be run 
    
                    
    kernel_gen: function that taken a batch of points, returns a kernel distr centered at those points
                in form of a torch.distribution
    prior: prior distribution from which the initial samples are drawn 
    burnin: number of discarded samples at the beginning of each markov chain
    returns: a tensor of shape (lenght - burnin, n_chains , distr_dims) of samples
    """
    s = torch.Size((n_chains,))
    x = prior.sample(s).squeeze(0)
    l = [x.double()]
    for i in range(lenght - 1):
        x = mh_step(x, log_energy, kernel_gen)
        l.append(x.clone())
    return torch.stack(l[burnin:], 0)

def so3_mh(log_energy, lenght, n_chains = 1, burnin = 0):
    return mh(log_energy, lenght, n_chains, so3_kernel_gen, SO3Prior(), burnin)
