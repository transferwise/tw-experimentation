from jax import lax
import jax.numpy as jnp
import jax.random as random

from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    is_prng_key,
    lazy_property,
    promote_shapes,
    validate_sample,
)
from numpyro.distributions import LogNormal


class ZeroInflatedProbsPatch(Distribution):
    """ZeroInflatedProbs distribution from Numpyro.

    https://num.pyro.ai/en/stable/_modules/numpyro/distributions/discrete.html#ZeroInflatedDistribution # noqa: E501

    Remove assertion check for base_dist.support.is_discrete to allow for
        continuous base_dist
    i.e. ZeroInflatedLogNormal
    """

    arg_constraints = {"gate": constraints.unit_interval}

    def __init__(self, base_dist, gate, *, validate_args=None):
        batch_shape = lax.broadcast_shapes(jnp.shape(gate), base_dist.batch_shape)
        (self.gate,) = promote_shapes(gate, shape=batch_shape)
        # assert base_dist.support.is_discrete
        if base_dist.event_shape:
            warning = "ZeroInflatedProbs expected empty base_dist.event_shape but got"
            raise ValueError(f"{warning} {base_dist.event_shape}")
        # XXX: we might need to promote parameters of base_dist but let's keep
        # this simplified for now
        self.base_dist = base_dist.expand(batch_shape)
        super(ZeroInflatedProbsPatch, self).__init__(
            batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        key_bern, key_base = random.split(key)
        shape = sample_shape + self.batch_shape
        mask = random.bernoulli(key_bern, self.gate, shape)
        samples = self.base_dist(rng_key=key_base, sample_shape=sample_shape)
        return jnp.where(mask, 0, samples)

    @validate_sample
    def log_prob(self, value):
        log_prob = jnp.log1p(-self.gate) + self.base_dist.log_prob(value)
        return jnp.where(value == 0, jnp.log(self.gate + jnp.exp(log_prob)), log_prob)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return self.base_dist.support

    @lazy_property
    def mean(self):
        return (1 - self.gate) * self.base_dist.mean

    @lazy_property
    def variance(self):
        return (1 - self.gate) * (
            self.base_dist.mean**2 + self.base_dist.variance
        ) - self.mean**2

    @property
    def has_enumerate_support(self):
        return self.base_dist.has_enumerate_support

    def enumerate_support(self, expand=True):
        return self.base_dist.enumerate_support(expand=expand)


class ZeroInflatedLogNormal(ZeroInflatedProbsPatch):
    arg_constraints = {
        "gate": constraints.unit_interval,
        "scale": constraints.positive,
        "loc": constraints.real,
    }
    support = constraints.real

    def __init__(self, gate=0.5, loc=0.0, scale=1.0, *, validate_args=None):
        _, self.loc, self.scale = promote_shapes(gate, loc, scale)
        super().__init__(
            LogNormal(loc=self.loc, scale=self.scale), gate, validate_args=validate_args
        )
