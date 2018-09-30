import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.transforms import Transform
from torch.distributions.utils import _sum_rightmost

from relie.flow import LocalDiffeoTransform


class LocalDiffeoTransformedDistribution(Distribution):
    r"""
    Version of TransformedDistribution that allows for non-injective maps
    with a discrete inverse set.
    """
    arg_constraints = {}

    def __init__(self, base_distribution, transforms, validate_args=None):
        self.base_dist = base_distribution
        if isinstance(transforms, Transform) or isinstance(
                transforms, LocalDiffeoTransform):
            self.transforms = [
                transforms,
            ]
        elif isinstance(transforms, list):
            if not all(
                    isinstance(t, Transform)
                    or isinstance(t, LocalDiffeoTransform)
                    for t in transforms):
                raise ValueError(
                    "transforms must be a Transform or a list of Transforms")
            self.transforms = transforms
        else:
            raise ValueError(
                "transforms must be a Transform or list, but was {}".format(
                    transforms))
        # TODO: Accommodate changes in shape
        shape = self.base_dist.batch_shape + self.base_dist.event_shape
        event_dim = max([len(self.base_dist.event_shape)] +
                        [t.event_dim for t in self.transforms])
        batch_shape = shape[:len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        return self.transforms[-1].codomain \
            if self.transforms else self.base_dist.support

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            for transform in self.transforms:
                x = transform(x)
            return x

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

    def log_prob(self, value):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        return self._log_prob(value, self.transforms)

    def _log_prob(self, y, transforms):
        event_dim = len(self.event_shape)
        if not transforms:
            return _sum_rightmost(
                self.base_dist.log_prob(y),
                event_dim - len(self.base_dist.event_shape))

        transform, *transforms = transforms

        if isinstance(transform, Transform):
            x = transform.inv(y)
            log_prob = -_sum_rightmost(
                transform.log_abs_det_jacobian(x, y),
                event_dim - transform.event_dim)
            return log_prob + self._log_prob(x, transforms)
        else:
            xset = transform.inverse_set(y)
            log_prob = -_sum_rightmost(
                transform.log_abs_det_jacobian(xset, y),
                event_dim - transform.event_dim)
            return torch.logsumexp(
                log_prob + self._log_prob(xset, transforms), dim=0)
