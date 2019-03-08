"""
Learn to fit a push-forward Gaussian with MLE.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import TensorDataset

from relie import LocalDiffeoTransformedDistribution, SO3ExpTransform
from relie.utils.data import TensorLoader, cycle
from relie.utils.so3_tools import so3_exp

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")


target_alg_distr = Normal(
    torch.tensor([0.0, 0.5, -0.5]).double(), torch.tensor([1.5, 0.1, 1.0]).double()
)
target_group_distr = LocalDiffeoTransformedDistribution(
    target_alg_distr, SO3ExpTransform(k_max=10)
)
data = so3_exp(target_alg_distr.sample((10000,)))
target_entropy = -target_group_distr.log_prob(data).mean()

dataset = TensorDataset(data)
loader = TensorLoader(dataset, 64, True)
loader_iter = cycle(loader)

loc = nn.Parameter(torch.zeros(3).double())
scale = nn.Parameter(torch.ones(3).double())
optimizer = torch.optim.Adam([loc, scale])

transform = SO3ExpTransform(k_max=10)

for it in range(100_000):
    batch, = next(loader_iter)
    alg_distr = Normal(loc * 1, scale * 1)
    distr = LocalDiffeoTransformedDistribution(alg_distr, transform)
    loss = -distr.log_prob(batch).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it % 1000 == 0:
        print(f"Loss: {loss.item()}. Global minimum: {target_entropy.item()}")
        print(f"Parameters: {torch.cat([loc, scale]).tolist()}")
