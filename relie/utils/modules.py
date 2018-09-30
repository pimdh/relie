import torch.nn as nn


class MLP(nn.Sequential):
    """Helper module to create MLPs."""
    def __init__(self, input_dims, output_dims, hidden_dims,
                 num_layers=1, activation=nn.ReLU):
        if num_layers == 0:
            super().__init__(nn.Linear(input_dims, output_dims))
        else:
            super().__init__(
                nn.Linear(input_dims, hidden_dims),
                activation(),
                *[l for _ in range(num_layers-1)
                  for l in [nn.Linear(hidden_dims, hidden_dims), activation()]],
                nn.Linear(hidden_dims, output_dims)
            )
