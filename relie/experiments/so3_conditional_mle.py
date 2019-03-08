"""
We have a dataset (x, G) for images x and orientations G:

Then we do MLE:
\max_f  E_{(x,G)}  [ \log p(G|x) ]
where
\log p(G|x) = \log \sum_{g \in Log(G)} N(g|mu(x), sigma(x))
for some prior  p_0

Data generation:
- We sample v \in R^d in some representation
- From uniform prior, we sample g \in G
- We act: v_g = g(v)
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import TensorDataset

from relie.flow import LocalDiffeoTransformedDistribution as LDTD, RadialTanhTransform
from relie.lie_distr import SO3ExpTransform, SO3Prior, SO3ExpCompactTransform
from relie.utils.data import TensorLoader, cycle
from relie.utils.so3_tools import so3_matrix_to_eazyz
from relie.utils.so3_rep_tools import block_wigner_matrix_multiply
from relie.utils.modules import MLP

device = torch.device("cpu")

# Prior distribution p(G)
# generation_alg_distr = Normal(torch.tensor([0., .5, -.5,]), torch.tensor([0.5, .1, 1.]))
# generation_group_distr = LDTD(generation_alg_distr, SO3ExpTransform())
generation_group_distr = SO3Prior(dtype=torch.double)

# Distribution to create noisy observations: p(G'|G)=n @ G, n \sim exp^*(N(0, .1))
noise_alg_distr = Normal(
    torch.tensor([0.0, 0.0, 0.0]).double(), torch.tensor([0.1, 0.1, 0.1]).double()
)
noise_group_distr = LDTD(noise_alg_distr, SO3ExpTransform())

# Sample true and noisy group actions
num_samples = 100_000
noise_samples = noise_group_distr.sample((num_samples,))
group_data = generation_group_distr.sample((num_samples,))
group_data_noised = noise_samples @ group_data

# Find global optimum liklihood (due to left-invariance this is correct)
target_entropy = -noise_group_distr.log_prob(noise_samples).mean()

# Create transformed data
angles = so3_matrix_to_eazyz(group_data)
max_rep_degree = 3
rep_copies = 1
x_dims = (max_rep_degree + 1) ** 2 * rep_copies
x_zero = torch.randn((max_rep_degree + 1) ** 2, rep_copies)
x_data = block_wigner_matrix_multiply(
    angles.float(), x_zero.expand(num_samples, -1, -1), max_rep_degree
)

dataset = TensorDataset(x_data, group_data_noised, group_data)
loader = TensorLoader(dataset, 64, True)
loader_iter = cycle(loader)


class ConditionalGaussianModel(nn.Module):
    """Models p(G|X) as exp^*(N(mu(x),sigma(X)))"""

    def __init__(self):
        super().__init__()
        self.module = MLP(x_dims, 2 * 3, 50, 8)

        # Set intital output at exp^*(N(0,1))
        last_module = list(self.module.modules())[-1]
        last_module.weight.data = torch.zeros_like(last_module.weight)
        last_module.bias.data = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    def distr(self, x):
        out = self.module(x.view(x.shape[0], -1)).double()
        loc, pre_scale = out[:, :3], out[:, 3:]
        alg_distr = Normal(loc, F.softplus(pre_scale))

        # Non-compact algebra region
        # transforms = SO3ExpTransform()

        # Compact algebra region
        transforms = [
            RadialTanhTransform(np.pi * 1.5),
            SO3ExpCompactTransform(np.pi * 1.5),
        ]
        return LDTD(alg_distr, transforms)

    def forward(self, x, g):
        log_prob = self.distr(x).log_prob(g)
        assert torch.isnan(log_prob).sum() == 0
        return -log_prob


model = ConditionalGaussianModel()
optimizer = torch.optim.Adam(model.parameters())


losses = []
for it in range(50000):
    x_batch, g_batch, _ = next(loader_iter)
    loss = model.forward(x_batch, g_batch).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if it % 1000 == 0:
        print(
            f"Loss: {np.mean(losses[-1000:]):.4f}. Global minimum: {target_entropy.item():.4f}"
        )


# %%
num_means = 1
num_noise_samples = 1000
means = generation_group_distr.sample((num_means,))

viz_data = []
for _ in range(num_means):
    mean = generation_group_distr.sample((1,))[0]
    x = block_wigner_matrix_multiply(
        so3_matrix_to_eazyz(mean[None].float()), x_zero[None], max_rep_degree
    )
    noise_samples = noise_group_distr.sample((num_noise_samples,))
    samples = (noise_samples @ mean[None]).view(num_noise_samples, 9)

    inferred_distr = model.distr(x)
    inferred_samples = inferred_distr.sample((num_noise_samples,)).view(
        num_noise_samples, 9
    )
    viz_data.append((samples, inferred_samples))

all_data = torch.cat([torch.cat(d) for d in viz_data])

pca = PCA(3).fit(all_data)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for i, (samples, inferred_samples) in enumerate(viz_data):
    samples = pca.transform(samples)
    inferred_samples = pca.transform(inferred_samples)
    ax.scatter(*samples.T, label=f"Original {i}", alpha=0.2)
    ax.scatter(*inferred_samples.T, label=f"Inferred {i}", alpha=0.2)

plt.legend()
plt.show()
