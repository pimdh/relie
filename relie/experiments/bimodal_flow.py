import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.distributions import TransformedDistribution, Normal
from relie.flow import CouplingTransform, PermuteTransform
from relie.utils.data import TensorLoader, cycle


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class CouplingFlowModel(nn.Module):
    """
    Note the reversed() and the .inv, this is because pytorch
    considers the Z -> X transformation, while the literature does the inverse.
    """
    def __init__(self, d, d_residue, n_layers):
        super().__init__()
        self.d = d
        self.nets = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(d_residue, 50),
                nn.ReLU(),
                nn.Linear(50, 50),
                nn.ReLU(),
                nn.Linear(50, (d-d_residue) * 2)
            ) for _ in range(n_layers)]
        )
        self.register_buffer('prior_loc', torch.zeros(d))
        self.register_buffer('prior_scale', torch.ones(d))

    def transforms(self):
        transforms = []
        for i, net in enumerate(self.nets):
            transforms.extend([
                CouplingTransform(1, net).inv,
                PermuteTransform([1, 0]).inv
            ])
        return transforms

    def distr(self):
        prior = Normal(self.prior_loc, self.prior_scale)
        return TransformedDistribution(prior, list(reversed(self.transforms())))

    def forward(self, x):
        return self.distr().log_prob(x).mean()


d = 2
num_samples = 1000
component_a = torch.randn(num_samples // 2, d) / 5 + torch.tensor([1., 1.])
component_b = torch.randn(num_samples // 2, d) / 5 + torch.tensor([-1., -1.])
data = torch.cat([component_a, component_b]).to(device)
dataset = TensorDataset(data)
loader = TensorLoader(dataset, 64, True)


model = CouplingFlowModel(d, 1, 6).to(device)
optimizer = torch.optim.Adam(model.parameters())

loader_iter = cycle(loader)
for it in range(100000):
    batch, = next(loader_iter)

    l = model(batch)
    optimizer.zero_grad()
    (-l).backward()
    optimizer.step()

    if it % 5000 == 0:
        print(l)
        samples = model.distr().sample((1000,))
        plt.scatter(*data.t(), s=2)
        plt.scatter(*samples.t(), s=2)
        plt.show()



