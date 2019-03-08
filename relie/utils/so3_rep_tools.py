"""
Tools for SO(3) representations.
"""
import torch


def _z_rot_mat(angle, l):
    m = angle.new_zeros((angle.size(0), 2 * l + 1, 2 * l + 1))

    inds = torch.arange(0, 2 * l + 1, 1, dtype=torch.long, device=angle.device)
    reversed_inds = torch.arange(2 * l, -1, -1, dtype=torch.long, device=angle.device)

    frequencies = torch.arange(l, -l - 1, -1, dtype=angle.dtype, device=angle.device)[
        None
    ]

    m[:, inds, reversed_inds] = torch.sin(frequencies * angle[:, None])
    m[:, inds, inds] = torch.cos(frequencies * angle[:, None])
    return m


class JContainer:
    data = {}

    @classmethod
    def get(cls, device):
        if str(device) in cls.data:
            return cls.data[str(device)]

        from lie_learn.representations.SO3.pinchon_hoggan.pinchon_hoggan_dense import (
            Jd as Jd_np,
        )

        device_data = [
            torch.tensor(J, dtype=torch.float32, device=device) for J in Jd_np
        ]
        cls.data[str(device)] = device_data

        return device_data


def wigner_d_matrix(angles, degree):
    """Create wigner D matrices for batch of ZYZ Euler anglers for degree l."""
    J = JContainer.get(angles.device)[degree][None]
    x_a = _z_rot_mat(angles[:, 0], degree)
    x_b = _z_rot_mat(angles[:, 1], degree)
    x_c = _z_rot_mat(angles[:, 2], degree)
    return x_a.matmul(J).matmul(x_b).matmul(J).matmul(x_c)


def block_wigner_matrix_multiply(angles, data, max_degree):
    """Transform data using wigner d matrices for all degrees.

    vector_dim is dictated by max_degree by the expression:
    vector_dim = \sum_{i=0}^max_degree (2 * max_degree + 1) = (max_degree+1)^2

    The representation is the direct sum of the irreps of the degrees up to max.
    The computation is equivalent to a block-wise matrix multiply.

    The data are the Fourier modes of a R^{data_dim} signal.

    Input:
    - angles (batch, 3)  ZYZ Euler angles
    - vector (batch, vector_dim, data_dim)

    Output: (batch, vector_dim, data_dim)
    """
    outputs = []
    start = 0
    for degree in range(max_degree + 1):
        dim = 2 * degree + 1
        matrix = wigner_d_matrix(angles, degree)
        outputs.append(matrix.bmm(data[:, start : start + dim, :]))
        start += dim
    return torch.cat(outputs, 1)
