import torch


def invariant_loss(x, y, symmetry):
    """
    Finds permutation invariant loss, for fixed set of allowed permutations.
    Computes picking minimal loss under all permutations.

    Uses L2 loss.
    :param x: Input of shape (..., d)
    :param y: Output of shape (..., d)
    :param symmetry: Symmetry transformation matrices of shape (n, d, d).
        Must include identity
    :return: Loss of shape (...)
    """
    batch_shape = x.shape[:-1]
    d = x.shape[-1]
    x = x.view(-1, d)  # [b, d]
    y = y.view(-1, d)  # [b, d]
    x_transformed = torch.einsum('nde,be->nbd', [symmetry, x])
    diff = (y - x_transformed).pow(2).sum(2)  # [n, b]
    min_diff = torch.min(diff, dim=0)[0]  # [b]
    return min_diff.view(batch_shape)
