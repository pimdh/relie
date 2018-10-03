"""
MLE of a multimodal distribution.
"""
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.distributions import Normal, ComposeTransform
from torch.utils.data import TensorDataset

from relie.flow import LocalDiffeoTransformedDistribution as LDTD,\
    PermuteTransform, CouplingTransform, RadialTanhTransform
from relie.lie_distr import SO3ExpTransform, SO3ExpCompactTransform
from relie.utils.data import TensorLoader, cycle
from relie.utils.so3_tools import so3_exp
from relie.utils.modules import MLP, ToTransform, BatchSqueezeModule

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Distribution to create noisy observations: p(G'|G)=n @ G, n \sim exp^*(N(0, .1))
noise_alg_distr = Normal(
    torch.zeros(3).double().to(device),
    torch.full((3, ), .1).double().to(device))
noise_group_distr = LDTD(noise_alg_distr, SO3ExpTransform())

# Sample true and noisy group actions
num_samples = 100000
noise_samples = noise_group_distr.sample((num_samples, ))

# Make symmetrical
symmetry_group_size = 3
symmetry_group = so3_exp(
    torch.tensor(
        [[2 * np.pi / symmetry_group_size * i, 0, 0]
         for i in range(symmetry_group_size)],
        device=device).double())

group_data = symmetry_group.repeat(num_samples // symmetry_group_size, 1, 1)
group_data_noised = noise_samples[:len(group_data)] @ group_data

dataset = TensorDataset(group_data_noised)
loader = TensorLoader(dataset, 640, True)
loader_iter = cycle(loader)


class Flow(nn.Module):
    def __init__(self, d, n_layers):
        super().__init__()
        self.d = d
        self.d_residue = 1
        self.d_transform = d - self.d_residue

        self.nets = nn.ModuleList([
            MLP(self.d_residue,
                2 * self.d_transform,
                50,
                3,
                batch_norm=False) for _ in range(n_layers)
        ])
        self._set_params()
        r = list(range(3))
        self.permutations = [r[i:] + r[:i] for i in range(3)]

    def forward(self):
        transforms = []
        for i, (net, permutation) in enumerate(
                zip(self.nets, cycle(self.permutations))):
            transforms.extend([
                CouplingTransform(self.d_residue, BatchSqueezeModule(net)),
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
            last_module.bias.data = torch.zeros_like(last_module.bias)


algebra_support_radius = np.pi * 1.1

intermediate_transform = ComposeTransform([
    RadialTanhTransform(algebra_support_radius),
    ToTransform(dict(dtype=torch.float32), dict(dtype=torch.float64))
])


class FlowDistr(nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow
        self.register_buffer('prior_loc', torch.zeros(3))
        self.register_buffer('prior_scale', torch.ones(3))

    def transforms(self):
        transforms = [
            self.flow().inv,
            intermediate_transform,
            SO3ExpCompactTransform(algebra_support_radius),
        ]
        return transforms

    def distr(self):
        prior = Normal(self.prior_loc, self.prior_scale)
        return LDTD(prior, self.transforms())

    def forward(self, g):
        log_prob = self.distr().log_prob(g)
        assert torch.isnan(log_prob).sum() == 0
        return -log_prob


flow_model = Flow(3, 12)
model = FlowDistr(flow_model).to(device)
optimizer = torch.optim.Adam(model.parameters())

losses = []
num_its = 3000
for it in range(num_its):
    g_batch, = next(loader_iter)
    loss = model.forward(g_batch).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    for n, p in model.named_parameters():
        assert torch.isnan(p).sum() == 0, f"NaN in parameters {n}"

    if it % 1000 == 0:
        print(f"Loss: {np.mean(losses[-1000:]):.4f}.")

# %%
model.eval()
num_noise_samples = 1000
inferred_distr = model.distr()
inferred_samples = inferred_distr.sample((num_noise_samples, )).view(-1, 9)
truth_samples = next(iter(loader))[0].view(-1, 9)
pca = PCA(3).fit(inferred_samples)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*pca.transform(inferred_samples).T, label="Model samples", alpha=.1)
ax.scatter(*pca.transform(truth_samples).T, label="Train data", alpha=.1)
ax.view_init(70, 30)
plt.legend()
plt.show()
