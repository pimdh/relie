# ReLie
Official repository to reproduce experiments for the AISTATS-19 publication on Reparameterizing Distributions over Lie Groups [1] ([arxiv](https://arxiv.org/abs/1903.02958)). For a more intuitive understanding of our work take a look at the [presentation slides prepared for our talk in Okinawa](https://github.com/pimdh/relie/blob/master/slides).

<p align="center"> 
<img src="https://i.imgur.com/WgKjQ7o.gif">
</p>

_From left to right, examples of SO(2), SO(3), and SE(3) group actions._


## Implementation
We implement the code for SO(3) in PyTorch by building on the `torch.distributions.transform` framework. We extend this framework, as the Lie group exponential map is not a bijection, but a locally invertible function / a local diffeomorphism:
- [`relie.LocalDiffeoTransform`](https://github.com/pimdh/relie/blob/master/relie/local_diffeo_transform.py) generalizes `torch.distributions.transform.Transform`
- [`relie.LocalDiffeoTransformedDistribution`](https://github.com/pimdh/relie/blob/master/relie/local_diffeo_transformed_distribution.py) generalizes `torch.distributions.transform.TransformedDistribution`

The simplest way of creating a distribution on the group, is by putting a zero-mean Gaussian on the algebra, pushing this forward and left-multiplying with a group element, to put the 'mean' of the resulting distribution away from the identity. This can be constructed as follows:
```
from relie import (
    SO3ExpTransform,
    SO3MultiplyTransform,
    LocalDiffeoTransformedDistribution as LDTD,
)

alg_loc = ...  # of shape [batch, 3], dtype=double
scale = ...  # of shape [batch, 3], dtype=double
loc = so3_exp(alg_loc)  # of shape [batch, 3, 3]

alg_distr = Normal(torch.zeros_like(scale), scale)
transforms = [SO3ExpTransform(k_max=3), SO3MultiplyTransform(loc)]
group_distr = LDTD(alg_distr, transforms)
```

This can then be used for e.g. Variational Inference:
```
z = group_distr.rsample()
entropy = -group_distr.log_prob(z)
```
Note:
- We require double precision.
- We consider `2 * k_max + 1` pre-images. In our experience, `k_max=3` is sufficient.
- Parametrizing the mean with an algebra element that is mapped to the group with the exponential map should not be used in the context of auto-encoders. See [2, 3] for details.


### LI-Flow
<p align="center"> 
<img src="https://i.imgur.com/8cls6fe.png">
</p>

Alternatively, one can construct a NICE-style normalizing flow. See [`relie.experiments.so3_multimodal_flow`](https://github.com/pimdh/relie/blob/master/relie/experiments/so3_multimodal_flow.py) for an example.

## Experiments
Please find the experiments of the paper in the package [`relie.experiments`](https://github.com/pimdh/relie/blob/master/relie/experiments).


## Contact
For comments and questions regarding this repository, feel free to reach out to [Pim de Haan](mailto:pimdehaan@gmail.com).

## License
MIT

## References
```
[1] Falorsi, L., de Haan, P., Davidson, T. & Forré, P.
Reparametrizing Distributions on Lie Groups
AISTATS (2019)
[2] Falorsi, L., de Haan, P., Davidson, T. R., De Cao, N., Weiler, M., Forré, P., & Cohen, T. S.
Explorations in homeomorphic variational auto-encoding
ICML 2018 workshop on Theoretical Foundations and Applications of Deep Generative Models (2018)
[3] de Haan, P., and Falorsi, L..
Topological Constraints on Homeomorphic Auto-Encoding
NeurIPS 2018 workshop on Integration of Deep Learning Theories (2018)
```

BibTeX format:
```
@article{falorsi2019reparameterizing,
  title={Reparameterizing distributions on Lie groups},
  author={Falorsi, L. and
          de Haan, P. and
          Davidson, T.R. and
          Forr{\'e}, P.},
  journal={22nd International Conference on Artificial Intelligence and Statistics (AISTATS-19)},
  year={2019}
}
```
