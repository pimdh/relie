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

from relie.flow import LocalDiffeoTransformedDistribution as LDTD, PermuteTransform, CouplingTransform, RadialTanhTransform, BatchNormTransform
from relie.lie_distr import SO3ExpTransform, SO3ExpCompactTransform, SO3Prior
from relie.utils.data import TensorLoader, cycle
from relie.utils.so3_tools import so3_matrix_to_eazyz, so3_exp
from relie.utils.so3_rep_tools import block_wigner_matrix_multiply
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
num_samples = 100000
noise_samples = noise_group_distr.sample((num_samples, ))
group_data = generation_group_distr.sample((num_samples, ))
group_data_noised = noise_samples @ group_data

# Create original data
max_rep_degree = 3
rep_copies = 1
x_dims = (max_rep_degree + 1)**2 * rep_copies
# x_zero = torch.randn((max_rep_degree + 1)**2, rep_copies, device=device)
x_zero = [
    0.9749450087547302, 1.320214033126831, 2.7400097846984863,
    0.6298772692680359, 0.8593451380729675, 0.6799159646034241,
    -0.4878562390804291, 1.2094298601150513, 0.009278437122702599,
    -1.52178156375885, 1.634827971458435, -1.2686760425567627,
    1.8586041927337646, 1.0522747039794922, -0.7130511403083801,
    0.3419789671897888
]
x_zero = torch.tensor(x_zero, device=device)[:, None]

# Make symmetrical
symmetry_group_size = 3
symmetry_group = so3_exp(
    torch.tensor(
        [[2 * np.pi / symmetry_group_size * i, 0, 0]
         for i in range(symmetry_group_size)],
        device=device).double())
x_data = block_wigner_matrix_multiply(
    so3_matrix_to_eazyz(symmetry_group).float(),
    x_zero.expand(symmetry_group_size, -1, -1), max_rep_degree)
x_zero = x_data.mean(0)

# Act with group
angles = so3_matrix_to_eazyz(group_data_noised)
x_data = block_wigner_matrix_multiply(angles.float(),
                                      x_zero.expand(num_samples, -1, -1),
                                      max_rep_degree)

dataset = TensorDataset(x_data, group_data, group_data)
loader = TensorLoader(dataset, 640, True)
loader_iter = cycle(loader)


class Flow(nn.Module):
    def __init__(self, d, d_conditional, n_layers):
        super().__init__()
        self.d = d
        self.d_residue = 1
        self.d_transform = d - self.d_residue
        self.x_preprocess = False
        self.batch_norm = True

        if self.x_preprocess:
            self.x_repr_dim = 9
            self.x_net = MLP(d_conditional, self.x_repr_dim, 50, 4, batch_norm=False)
        else:
            self.x_repr_dim = d_conditional
            self.x_net = None

        self.nets = nn.ModuleList([
            MLP(self.d_residue + self.x_repr_dim,
                2 * self.d_transform,
                50,
                3,
                batch_norm=False) for _ in range(n_layers)
        ])
        self._set_params()
        r = list(range(3))
        self.permutations = [r[i:] + r[:i] for i in range(3)]

        if self.batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(d) for _ in range(n_layers)])
        else:
            self.batch_norms = [None] * n_layers

    def forward(self, x):
        if self.x_net is not None:
            x = self.x_net(x)
        transforms = []
        for i, (net, bn, permutation) in enumerate(
                zip(self.nets, self.batch_norms, cycle(self.permutations))):
            if bn is not None:
                transforms.append(BatchNormTransform(bn))
            transforms.extend([
                CouplingTransform(self.d_residue, ConditionalModule(net, x)),
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


algebra_support_radius = np.pi * 1.6

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

    def transforms(self, x):
        transforms = [
            self.flow(x).inv,
            intermediate_transform,
            SO3ExpCompactTransform(algebra_support_radius),
        ]
        return transforms

    def distr(self, x):
        prior = Normal(self.prior_loc, self.prior_scale)
        return LDTD(prior, self.transforms(x))

    def forward(self, x, g):
        log_prob = self.distr(x).log_prob(g)
        assert torch.isnan(log_prob).sum() == 0
        return -log_prob


flow_model = Flow(3, x_dims, 12)
model = FlowDistr(flow_model).to(device)
# model.load_state_dict(torch.load('models/so3-multimodal-1538530160.pkl'))
optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)

losses = []
num_its = 20000
for it in range(num_its):
    x_batch, g_batch, g_truth = next(loader_iter)
    x_batch = x_batch.view(-1, x_dims)
    loss = model.forward(x_batch, g_batch).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    for n, p in model.named_parameters():
        assert torch.isnan(p).sum() == 0, f"NaN in parameters {n}"

    if it % 1000 == 0:
        print(f"It {it}. Loss: {np.mean(losses[-1000:]):.4f}.")

if num_its > 0:
    path = os.path.join('models', f'so3-multimodal-{int(time())}.pkl')
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# %%
model.eval()
# model.load_state_dict(torch.load('models/so3-multimodal-1538530160.pkl'))
for _ in range(5):
    num_noise_samples = 1000
    g_zero = generation_group_distr.sample((1, ))[0]
    g_subgroup = symmetry_group @ g_zero[None]
    # g_subgroup = g_zero[None]
    x = block_wigner_matrix_multiply(
        so3_matrix_to_eazyz(g_zero[None].float()), x_zero[None], max_rep_degree)

    inferred_distr = model.distr(x.view(-1).expand(num_noise_samples, -1))
    samples = inferred_distr.sample((num_noise_samples, )).view(
        num_noise_samples, 9)
    # pca = PCA(3).fit(g_subgroup.view(4, 9))
    pca = PCA(3).fit(samples)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*pca.transform(samples).T, label="Inferred", alpha=.2)
    ax.scatter(
        *pca.transform(g_subgroup.view(-1, 9)).T,
        label="Ground truth",
        s=100,
        alpha=1,
        zorder=2)
    ax.view_init(70, 30)
    plt.legend()
    plt.show()

