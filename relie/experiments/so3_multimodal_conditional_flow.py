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
import argparse
import logging
import numpy as np
from relie.utils.experiment import setup_experiment, tensor_read_image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.utils import Bunch

import torch
import torch.nn as nn
from torch.distributions import Normal, ComposeTransform
from torch.utils.data import TensorDataset

from relie.flow import (
    LocalDiffeoTransformedDistribution as LDTD,
    PermuteTransform,
    CouplingTransform,
    RadialTanhTransform,
    BatchNormTransform,
)
from relie.lie_distr import SO3ExpTransform, SO3ExpCompactTransform, SO3Prior
from relie.utils.data import TensorLoader, cycle
from relie.utils.so3_tools import so3_matrix_to_eazyz, so3_exp
from relie.utils.so3_rep_tools import block_wigner_matrix_multiply
from relie.utils.modules import MLP, ConditionalModule, ToTransform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Flow(nn.Module):
    def __init__(
        self,
        d,
        d_conditional,
        n_layers,
        batch_norm=True,
        x_preprocess=False,
        net_layers=3,
    ):
        super().__init__()
        self.d = d
        self.d_residue = 1
        self.d_transform = d - self.d_residue

        if x_preprocess:
            self.x_repr_dim = 9
            self.x_net = MLP(d_conditional, self.x_repr_dim, 50, 4, batch_norm=False)
        else:
            self.x_repr_dim = d_conditional
            self.x_net = None

        self.nets = nn.ModuleList(
            [
                MLP(
                    self.d_residue + self.x_repr_dim,
                    2 * self.d_transform,
                    50,
                    net_layers,
                    batch_norm=False,
                )
                for _ in range(n_layers)
            ]
        )
        self._set_params()
        r = list(range(3))
        self.permutations = [r[i:] + r[:i] for i in range(3)]

        if batch_norm:
            self.batch_norms = nn.ModuleList(
                [nn.BatchNorm1d(d) for _ in range(n_layers)]
            )
        else:
            self.batch_norms = [None] * n_layers

    def forward(self, x):
        if self.x_net is not None:
            x = self.x_net(x)
        transforms = []
        for i, (net, bn, permutation) in enumerate(
            zip(self.nets, self.batch_norms, cycle(self.permutations))
        ):
            if bn is not None:
                transforms.append(BatchNormTransform(bn))
            transforms.extend(
                [
                    CouplingTransform(self.d_residue, ConditionalModule(net, x)),
                    PermuteTransform(permutation),
                ]
            )
        return ComposeTransform(transforms)

    def _set_params(self):
        """
        Initialize coupling layers to be identity.
        """
        for net in self.nets:
            last_module = list(net.modules())[-1]
            last_module.weight.data = torch.zeros_like(last_module.weight)
            last_module.bias.data = torch.zeros_like(last_module.bias)


class FlowDistr(nn.Module):
    def __init__(self, flow, algebra_support_radius=np.pi * 1.6):
        super().__init__()
        self.flow = flow
        self.register_buffer("prior_loc", torch.zeros(3))
        self.register_buffer("prior_scale", torch.ones(3))
        self.intermediate_transform = ComposeTransform(
            [
                RadialTanhTransform(algebra_support_radius),
                ToTransform(dict(dtype=torch.float32), dict(dtype=torch.float64)),
            ]
        )
        self.algebra_support_radius = algebra_support_radius

    def transforms(self, x):
        transforms = [
            self.flow(x).inv,
            self.intermediate_transform,
            SO3ExpCompactTransform(self.algebra_support_radius),
        ]
        return transforms

    def distr(self, x):
        prior = Normal(self.prior_loc, self.prior_scale)
        return LDTD(prior, self.transforms(x))

    def forward(self, x, g):
        log_prob = self.distr(x).log_prob(g)
        assert torch.isnan(log_prob).sum() == 0
        return -log_prob


def gen_data(symmetry_group_size=3, noise=0.1, num_samples=100_000):
    # Prior distribution p(G)
    generation_group_distr = SO3Prior(dtype=torch.double, device=device)

    # Distribution to create noisy observations: p(G'|G)=n @ G, n \sim exp^*(N(0, .1))
    noise_alg_distr = Normal(
        torch.zeros(3).double().to(device), torch.full((3,), noise).double().to(device)
    )
    noise_group_distr = LDTD(noise_alg_distr, SO3ExpTransform())

    # Sample true and noisy group actions
    noise_samples = noise_group_distr.sample((num_samples,))
    group_data = generation_group_distr.sample((num_samples,))
    group_data_noised = noise_samples @ group_data

    # Create original data
    max_rep_degree = 3
    rep_copies = 1
    x_dims = (max_rep_degree + 1) ** 2 * rep_copies
    # x_zero = torch.randn((max_rep_degree + 1)**2, rep_copies, device=device)
    x_zero = [
        0.974_945_008_754_730_2,
        1.320_214_033_126_831,
        2.740_009_784_698_486_3,
        0.629_877_269_268_035_9,
        0.859_345_138_072_967_5,
        0.679_915_964_603_424_1,
        -0.487_856_239_080_429_1,
        1.209_429_860_115_051_3,
        0.009_278_437_122_702_599,
        -1.521_781_563_758_85,
        1.634_827_971_458_435,
        -1.268_676_042_556_762_7,
        1.858_604_192_733_764_6,
        1.052_274_703_979_492_2,
        -0.713_051_140_308_380_1,
        0.341_978_967_189_788_8,
    ]
    x_zero = torch.tensor(x_zero, device=device)[:, None]

    # Make symmetrical
    symmetry_group = so3_exp(
        torch.tensor(
            [
                [2 * np.pi / symmetry_group_size * i, 0, 0]
                for i in range(symmetry_group_size)
            ],
            device=device,
        ).double()
    )
    x_data = block_wigner_matrix_multiply(
        so3_matrix_to_eazyz(symmetry_group).float(),
        x_zero.expand(symmetry_group_size, -1, -1),
        max_rep_degree,
    )
    x_zero = x_data.mean(0)

    # Act with group
    angles = so3_matrix_to_eazyz(group_data_noised)
    x_data = block_wigner_matrix_multiply(
        angles.float(), x_zero.expand(num_samples, -1, -1), max_rep_degree
    )

    dataset = TensorDataset(x_data, group_data, group_data)
    loader = TensorLoader(dataset, 640, True)
    loader_iter = cycle(loader)
    return Bunch(
        loader_iter=loader_iter,
        x_dims=x_dims,
        generation_group_distr=generation_group_distr,
        symmetry_group=symmetry_group,
        x_zero=x_zero,
        max_rep_degree=max_rep_degree,
    )


def plot(model, data, out_path, tb_writer, it):
    model.eval()
    for i in range(5):
        num_noise_samples = 1000
        g_zero = data.generation_group_distr.sample((1,))[0]
        g_subgroup = data.symmetry_group @ g_zero[None]
        # g_subgroup = g_zero[None]
        x = block_wigner_matrix_multiply(
            so3_matrix_to_eazyz(g_zero[None].float()),
            data.x_zero[None],
            data.max_rep_degree,
        )

        inferred_distr = model.distr(x.view(-1).expand(num_noise_samples, -1))
        samples = inferred_distr.sample((num_noise_samples,)).view(num_noise_samples, 9)
        mask = np.isfinite(samples).any(dim=1)
        if np.logical_not(mask).any():
            fails = np.logical_not(mask).sum()
            print(f"Failed to sample {fails} elements, filtering")
            samples = samples[mask]

        # pca = PCA(3).fit(g_subgroup.view(4, 9))
        pca = PCA(3).fit(samples)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(*pca.transform(samples).T, label="Inferred", alpha=0.2)
        ax.scatter(
            *pca.transform(g_subgroup.view(-1, 9)).T,
            label="Ground truth",
            s=100,
            alpha=1,
        )
        ax.view_init(70, 30)
        plt.legend()
        path = out_path(category=f"imgs/{it}", filename=f"{i}.png")
        plt.savefig(path)
        tb_writer.add_image(f"samples-{i}", tensor_read_image(path), it)
        plt.show()
        plt.close()


def checkpoint(model, optimizer, path):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)


def restore(model, optimizer, path):
    data = torch.load(path)
    if "model" in data:
        model.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optimizer"])
    else:  # Backwards compatible
        model.load_state_dict(data)


def main():
    parser = argparse.ArgumentParser("SO(3) multimodal conditional flow")
    parser.add_argument("--name")
    parser.add_argument("--flow_layers", type=int, default=18)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_its", type=int, default=50000)
    parser.add_argument("--num_samples", type=int, default=100_000)
    parser.add_argument("--load_path")
    args = parser.parse_args()

    tb_writer, out_path = setup_experiment("flow", args.name, args)

    data = gen_data(noise=args.noise, num_samples=args.num_samples)
    flow_model = Flow(3, data.x_dims, args.flow_layers)
    model = FlowDistr(flow_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.load_path:
        restore(model, optimizer, args.load_path)

    losses = []
    for it in range(args.num_its):
        model.train()
        x_batch, g_batch, g_truth = next(data.loader_iter)
        x_batch = x_batch.view(-1, data.x_dims)
        loss = model.forward(x_batch, g_batch).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        for n, p in model.named_parameters():
            assert torch.isnan(p).sum() == 0, f"NaN in parameters {n}"

        if it % 1000 == 0:
            logging.info(f"It {it}. Loss: {np.mean(losses[-1000:]):.4f}.")
            tb_writer.add_scalar("loss", np.mean(losses[-1000:]), it)

            save_path = out_path(filename="model.pkl")
            checkpoint(model, optimizer, save_path)
            if it % 5000 == 0:
                plot(model, data, out_path, tb_writer, it)

    logging.info(f"Model saved to {save_path}")
    plot(model, data, out_path, tb_writer, args.num_its)


if __name__ == "__main__":
    main()
