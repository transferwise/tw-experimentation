import numpyro.distributions as dist
from numpyro.distributions import (
    ZeroInflatedPoisson,
    BernoulliProbs,
    LogNormal,
    Gamma,
    Uniform,
    Beta,
    constraints,
    Normal,
    Exponential,
    Distribution,
)
from tw_experimentation.constants import MetricType

from tw_experimentation.bayes.numpyro_monkeypatch import ZeroInflatedLogNormal
from abc import ABCMeta, abstractmethod


# AVAILABLE_LIKELIHOOD_MODELS = {
#     "binary": [Bernoulli],
#     "continuous": [ZeroInflatedLogNormal, Normal, Exponential],
#     "discrete": [ZeroInflatedPoisson],
# }


# LIKELIHOOD_PRIORS = {
#     "Bernoulli": {"probs": Beta},
#     "ZeroInflatedLogNormal": {"loc": LogNormal, "scale": Gamma, "gate": Uniform},
#     "Normal": {"loc": LogNormal, "scale": Gamma},
#     "ZeroInflatedPoisson": {"rate": Gamma, "gate": Beta},
#     "Exponential": {"rate": Gamma},
# }

# EXPECTED_VALUE_PER_LIKELIHOOD = {}

# PRIOS_PARAMS = {
#     "LogNormal": {"args": {}, "kwargs": {"loc": 0, "scale": 1}},
#     "Gamma": {"args": (1,), "kwargs": {"rate": 0.5}},
#     "Uniform": {"args": (), "kwargs": {"low": 0, "high": 1}},
#     "Beta": {"args": (2, 2), "kwargs": {}},
#     "Gamma": {"args": (1,), "kwargs": {"rate": 0.5}},
#     "Beta": {"args": (2, 2), "kwargs": {}},
# }

# class Likelihood(ABCMeta):
#     @property
#     @abstractmethod
#     def metrictype(self):
#         raise NotImplementedError()

#     @property
#     @abstractmethod
#     def auxiliary_zero_inflation(self):
#         raise NotImplementedError()


class BernoulliLikelihood(BernoulliProbs):
    def metrictype(self):
        return MetricType.BINARY

    @property
    def auxiliary_zero_inflation(self):
        return False

    @property
    def aux_dist(self):
        if not self.auxiliary_zero_inflation:
            return None
        else:
            return Uniform(low=0, high=1)


class ZeroInflatedLogNormalLikelihood(LogNormal):
    def metrictype(self):
        return MetricType.CONTINUOUS

    @property
    def auxiliary_zero_inflation(self):
        return True

    def aux_dist(self):
        if not self.auxiliary_zero_inflation:
            return None
        else:
            return Uniform


class ZeroInflatedPoissonLikelihood(ZeroInflatedPoisson):
    def metrictype(self):
        return MetricType.DISCRETE

    @property
    def auxiliary_zero_inflation(self):
        return False

    @property
    def aux_dist(self):
        if not self.auxiliary_zero_inflation:
            return None
        else:
            return Uniform


DEFAULT_LIKELIHOOD_MODELS = {
    "binary": BernoulliLikelihood,
    "continuous": ZeroInflatedLogNormalLikelihood,
    "discrete": ZeroInflatedPoissonLikelihood,
}


DEFAULT_PRIOR_MODELS = {
    "binary": {"probs": Beta},
    "continuous": {
        "loc": Normal,
        "scale": Gamma,
        # "gate": Uniform
    },
    "discrete": {"rate": Gamma, "gate": Beta},
}


DEFAULT_PRIOR_MODEL_PARAMS = {
    "binary": {"probs": {"args": (2, 2), "kwargs": {}}},
    "continuous": {
        "loc": {"args": {}, "kwargs": {"loc": 0, "scale": 0.25}},
        "scale": {"args": (0.5,), "kwargs": {"rate": 0.5}},
        # "gate": {"args": (), "kwargs": {"low": 0, "high": 1}},
    },
    "discrete": {
        "rate": {"args": (1,), "kwargs": {"rate": 0.5}},
        "gate": {"args": (2, 2), "kwargs": {}},
    },
}


# class BernoulliLikelihood(Likelihood, Bernoulli):
#     def metrictype(self):
#         return MetricType.BINARY

#     def param_names(self):
#         return {"probs": BetaPrior()}

#     def expected_value(self):
#         return "probs"


# class ZeroInflatedPoissonLikelihood(Likelihood, ZeroInflatedPoisson):
#     def metrictype(self):
#         return MetricType.DISCRETE


# class ZeroInflatedLogNormalLikelihood(Likelihood, ZeroInflatedLogNormal):
#     def metrictype(self):
#         return MetricType.CONTINUOUS


# class Prior:
#     @property
#     def args(self):
#         raise NotImplementedError()

#     @property
#     def kwargs(self):
#         raise NotImplementedError()


# class BetaPrior(Prior, Beta):
#     def args(self):
#         return (2, 2)

#     def kwargs(self):
#         return {}
