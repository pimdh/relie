"""
We have a dataset (x, G) for images x and orientations G:
We learn a map  f:  X x R^d -> R^d, such that for any x, f_x  : R^d -> R^d is invertible.

Then we do MLE:
\max_f  E_{(x,G)}  [ \log p(G|x) ]
where
\log p(G|x) = \log \sum_{g \in Log(G)} p_0(f(x, g)) + change of variables
for some prior  p_0

Data generation:
- We sample v \in R^d in some representation
- We make this symmetrical by acting v = v + g_1(v) + ... for g_i in discrete subgroup
- From uniform prior, we sample g \in G
- We act: v_g = g(v)
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.distributions import Normal, ComposeTransform, SigmoidTransform, AffineTransform
from torch.utils.data import TensorDataset

from relie.flow import LocalDiffeoTransformedDistribution as LDTD, PermuteTransform, CouplingTransform
from relie.lie_distr import SO3ExpTransform, SO3ExpCompactTransform, SO3Prior
from relie.utils.data import TensorLoader, cycle
from relie.utils.so3_tools import so3_matrix_to_eazyz, block_wigner_matrix_multiply, so3_exp
from relie.utils.modules import MLP, ConditionalModule, ToTransform

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Prior distribution p(G)
generation_group_distr = SO3Prior(dtype=torch.double, device=device)

# Distribution to create noisy observations: p(G'|G)=n @ G, n \sim exp^*(N(0, .1))
noise_alg_distr = Normal(
    torch.zeros(3).double().to(device),
    torch.full((3, ), .1).double().to(device))
noise_group_distr = LDTD(noise_alg_distr, SO3ExpTransform())

# Sample true and noisy group actions
num_samples = 10000
noise_samples = noise_group_distr.sample((num_samples, ))
group_data = generation_group_distr.sample((num_samples, ))
group_data_noised = noise_samples @ group_data

# Create original data
max_rep_degree = 3
rep_copies = 1
x_dims = (max_rep_degree + 1)**2 * rep_copies
x_zero = torch.randn((max_rep_degree + 1)**2, rep_copies, device=device)

# Make symmetrical
symmetry_group = so3_exp(
    torch.tensor([[np.pi / 2 * i, 0, 0] for i in range(4)], device=device).double())
x_data = block_wigner_matrix_multiply(
    so3_matrix_to_eazyz(symmetry_group).float(), x_zero.expand(4, -1, -1),
    max_rep_degree)
x_zero = x_data.mean(0)

# Act with group
angles = so3_matrix_to_eazyz(group_data)
x_data = block_wigner_matrix_multiply(angles.float(),
                                      x_zero.expand(num_samples, -1, -1),
                                      max_rep_degree)

dataset = TensorDataset(x_data, group_data_noised, group_data)
loader = TensorLoader(dataset, 64, True)
loader_iter = cycle(loader)


class Flow(nn.Module):
    def __init__(self, d, d_conditional, n_layers):
        super().__init__()
        self.d = d
        self.d_residue = 2
        self.d_transform = 1
        self.nets = nn.ModuleList([
            MLP(self.d_residue + d_conditional, 2 * self.d_transform, 50, 4)
            for _ in range(n_layers)
        ])
        self._set_params()
        r = list(range(3))
        self.permutations = [r[i:] + r[:i] for i in range(3)]

    def forward(self, x):
        transforms = []
        for i, (net, permutation) in enumerate(
                zip(self.nets, cycle(self.permutations))):
            transforms.extend([
                CouplingTransform(2, ConditionalModule(net, x)),
                PermuteTransform(permutation),
            ])
        return ComposeTransform(transforms)

    def _set_params(self):
        """
        Initialize coupling layers to be identity.
        """
        for net in self.nets:
            last_module = list(net.modules())[-1]
            last_module.weight.data = torch.zeros_like(last_module.weight)
            last_module.bias.data = torch.tensor([0., 0.])


intermediate_transform = ComposeTransform([
    SigmoidTransform(1),  # (-inf, inf)->(0, 1)
    AffineTransform(-2 * np.pi, 4 * np.pi,
                    cache_size=1),  # (0, 1)->(-2pi, 2pi)
    ToTransform(dict(dtype=torch.float32), dict(dtype=torch.float64))
])


class FlowDistr(nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow
        self.register_buffer('prior_loc', torch.zeros(3))
        self.register_buffer('prior_scale', torch.ones(3))

    def transforms(self, x):
        transforms = [
            self.flow(x).inv,
            intermediate_transform,
            SO3ExpCompactTransform(),
        ]
        return transforms

    def distr(self, x):
        prior = Normal(self.prior_loc, self.prior_scale)
        return LDTD(prior, self.transforms(x))

    def forward(self, x, g):
        log_prob = self.distr(x).log_prob(g)
        assert torch.isnan(log_prob).sum() == 0
        return -log_prob


flow_model = Flow(6, x_dims, 9)
model = FlowDistr(flow_model).to(device)
optimizer = torch.optim.Adam(model.parameters())

losses = []
for it in range(20000):
    x_batch, g_batch, _ = next(loader_iter)
    x_batch = x_batch.view(-1, x_dims)
    loss = model.forward(x_batch, g_batch).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if it % 1000 == 0:
        print(f"Loss: {np.mean(losses[-1000:]):.4f}.")

# %%
num_noise_samples = 1000
g_zero = generation_group_distr.sample((1,))[0]
g_subgroup = symmetry_group @ g_zero[None]
x = block_wigner_matrix_multiply(
    so3_matrix_to_eazyz(g_zero[None].float()), x_zero[None], max_rep_degree)

inferred_distr = model.distr(x.view(-1).expand(num_noise_samples, -1))
samples = inferred_distr.sample((num_noise_samples,)).view(num_noise_samples, 9)
pca = PCA(3).fit(g_subgroup.view(4, 9))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*pca.transform(samples).T, label="Inferred", alpha=.5)
ax.scatter(*pca.transform(g_subgroup.view(4, 9)).T, label="Ground truth", s=50)
plt.legend()
plt.show()
