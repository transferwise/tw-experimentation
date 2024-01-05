from tw_experimentation.utils import ExperimentDataset
from tw_experimentation.data_generation import *
from tw_experimentation.constants import MetricType

import pandas as pd
import numpy as np


from typing import List

import numpyro
import numpyro.distributions as dist


from jax import random
from jax import numpy as jnp
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import Predictive


from tw_experimentation.bayes.bayes_params import (
    DEFAULT_LIKELIHOOD_MODELS,
    DEFAULT_PRIOR_MODELS,
    DEFAULT_PRIOR_MODEL_PARAMS,
)


class BayesModel:
    def __init__(
        self,
        metric_type: str,
        observations: pd.core.groupby.generic.DataFrameGroupBy,
        n_variants: int = 2,
        do_bayesian_updating: bool = True,
    ) -> None:
        self.metric_type = metric_type
        self.do_bayesian_updating = do_bayesian_updating
        self.observations = {}
        for variant_name, variant_observations in observations:
            self.observations[str(variant_name)] = variant_observations.to_numpy()

        self.n_variants = n_variants

        assert len(self.observations) == self.n_variants

        available_types = [metric_type.value for metric_type in MetricType]
        assert (
            metric_type in available_types
        ), "Metric types must be one of 'binary', 'discrete', or 'continuous'"

        self.likelihood_model = DEFAULT_LIKELIHOOD_MODELS[self.metric_type]

        self.prior_model = DEFAULT_PRIOR_MODELS[self.metric_type]
        self.prior_model_params = DEFAULT_PRIOR_MODEL_PARAMS[self.metric_type]

    def set_model(
        self,
        likelihood_model,
        variables: List[str],
        prior_models: List,
        params_models: List[dict],
    ):
        self.set_likelihood_model(likelihood_model)

        # reset prior_modle and prior_model_params
        self.prior_model = {}
        self.prior_model_params = {}

        for model_and_params in zip(variables, prior_models, params_models):
            self.set_prior_model(*model_and_params)

        assert self._model_is_well_defined()

    def set_prior_model(self, variable: str, model, model_params: dict):
        """Set prior model for a likelihood model parameter

        Args:
            variable (str): name of variab model variable to be fed into likelihood
            model (...): numpyro distribution used as prior
            model_params (dict): args and kwargs for prior model
                                Must be of form
                                {'args': (a,b),
                                 'kwargs': {'c': d}
                                 }
        """
        self.prior_model[variable] = model
        self.prior_model_params[variable] = model_params

    def set_prior_model_param(self, variable: str, model_params: dict):
        """Set parameters for prior

        Args:
            variable (str):     name of variab model variable to be fed into likelihood
            model_params (dict): args and kwargs for prior model
                                Must be of form
                                {'args': (a,b),
                                 'kwargs': {'c': d}
                                 }
        """
        self.prior_model_params[variable] = model_params

    def set_likelihood_model(self, model):
        self.likelihood_model = model

    def build_model(self):
        models = {}
        deltas = {}
        ate = {}
        means = {}
        aux_dist = {}

        for k, obs in self.observations.items():
            models[k] = {}

            if int(k) > 0:  # treatment to control comparison
                deltas[k] = {}

            for likelihood_param, distribution in self.prior_model.items():
                models[k][likelihood_param] = numpyro.sample(
                    f"variant_{k}_{likelihood_param}",
                    distribution(
                        *self.prior_model_params[likelihood_param]["args"],
                        **self.prior_model_params[likelihood_param]["kwargs"],
                    ),
                )

                if int(k) > 0:  # treatment to control comparison
                    deltas[k][likelihood_param] = numpyro.deterministic(
                        f"delta_{k}_{likelihood_param}",
                        models[k][likelihood_param] - models["0"][likelihood_param],
                    )

            if self.likelihood_model.auxiliary_zero_inflation:
                aux_dist[k] = numpyro.sample(
                    f"aux_dist_{k}", dist.Uniform(low=0, high=1)
                )
                if self.do_bayesian_updating:
                    with numpyro.plate(f"likelihood_{k}", obs[np.nonzero(obs)].size):
                        numpyro.sample(
                            f"likelihood_model_{k}",
                            self.likelihood_model(**models[k]),
                            obs=obs[np.nonzero(obs)],
                        )
                    with numpyro.plate(f"posterior_gate_{k}", obs.size):
                        numpyro.sample(
                            f"posterior_gate_model_{k}",
                            dist.Bernoulli(probs=aux_dist[k]),
                            obs=jnp.where(obs > 0, 1, 0),
                        )

            else:
                if self.do_bayesian_updating:
                    with numpyro.plate(f"likelihood_{k}", obs.size):
                        numpyro.sample(
                            f"likelihood_model_{k}",
                            self.likelihood_model(**models[k]),
                            obs=obs,
                        )

            if int(k) > 0:
                # ate[k] = numpyro.deterministic(
                #     f"ate_{k}",
                #     self.posterior_ate(
                #         models,
                #         k,
                #         auxiliary_zero_inflation=self.likelihood_model.auxiliary_zero_inflation,
                #         aux_dist=aux_dist,
                #     ),
                # )

                ate[k] = numpyro.deterministic(
                    f"ate_{k}",
                    self.posterior_ate(
                        models,
                        k,
                        auxiliary_zero_inflation=self.likelihood_model.auxiliary_zero_inflation,
                        aux_dist=aux_dist,
                    ),
                )

            means[k] = numpyro.deterministic(
                f"means_{k}",
                self.posterior_means(
                    models,
                    k,
                    auxiliary_zero_inflation=self.likelihood_model.auxiliary_zero_inflation,
                    aux_dist=aux_dist,
                ),
            )

    def posterior_ate(
        self, numpyro_model, treatment, auxiliary_zero_inflation=False, aux_dist=None
    ):
        if auxiliary_zero_inflation:
            assert aux_dist is not None

        add_zeros_t = aux_dist[treatment] if auxiliary_zero_inflation else 1
        add_zeros_c = aux_dist["0"] if auxiliary_zero_inflation else 1
        if self.metric_type == "binary":
            ate = numpyro_model[treatment]["probs"] - numpyro_model["0"]["probs"]
        elif self.metric_type == "continuous":
            ate = (
                jnp.exp(
                    numpyro_model[treatment]["loc"]
                    + numpyro_model[treatment]["scale"] ** 2 / 2
                )
                * add_zeros_t
            ) - (
                jnp.exp(
                    numpyro_model["0"]["loc"] + numpyro_model["0"]["scale"] ** 2 / 2
                )
                * add_zeros_c
            )
        elif self.metric_type == "discrete":
            ate = numpyro_model[treatment]["rate"] * numpyro_model[treatment][
                "gate"
            ] - (numpyro_model["0"]["rate"] * numpyro_model["0"]["gate"])
        return ate

    def posterior_means(
        self, numpyro_model, variant, auxiliary_zero_inflation=False, aux_dist=None
    ):
        add_zeros = aux_dist[variant] if auxiliary_zero_inflation else 1
        if self.metric_type == "binary":
            return numpyro_model[variant]["probs"]
        elif self.metric_type == "continuous":
            return (
                jnp.exp(
                    numpyro_model[variant]["loc"]
                    + numpyro_model[variant]["scale"] ** 2 / 2
                )
                * add_zeros
            )
        elif self.metric_type == "discrete":
            return numpyro_model[variant]["rate"] * numpyro_model[variant]["gate"]

    def _model_is_well_defined(self):
        # check whether parameters needed for likelihood model all appear in self.prior_model
        return True

    def get_likelihood_variables(self):
        return list(self.prior_model.keys())


# for debugging
if __name__ == "__main__":
    from statistical_tests import *

    rc = RevenueConversion()
    df = rc.generate_data()

    targets = ["conversion", "revenue"]
    metrics = ["binary", "continuous"]

    ed = ExperimentDataset(
        data=df,
        variant="T",
        targets=targets,
        date="trigger_dates",
        metric_types=dict(zip(targets, metrics)),
    )

    ed.preprocess_dataset()

    ### Test 1
    bm = BayesModel(
        metric_type="binary", observations=ed.data.groupby("T")["conversion"]
    )
    nuts = NUTS(bm.build_model, adapt_step_size=True)
    mcmc = MCMC(nuts, num_samples=1000, num_warmup=500)
    rng_key = random.PRNGKey(0)

    bm.set_prior_model_param("probs", {"args": (5, 4), "kwargs": {}})

    mcmc.run(
        rng_key,
    )
    mcmc.print_summary(exclude_deterministic=False)

    sam = mcmc.get_samples()
    pred = Predictive(bm.build_model, num_samples=1000)
    y_pred = pred(rng_key)["likelihood_model_0"]
    pred_post = Predictive(
        bm.build_model, posterior_samples=sam, num_samples=1000, batch_ndims=2
    )
    y_post_pred_0 = pred_post(rng_key)["likelihood_model_0"]
    y_post_pred_1 = pred_post(rng_key)["likelihood_model_1"]
    y_post_all = pred_post(rng_key)
    n_points = min(y_post_pred_0.shape[1], y_post_pred_0.shape[1])

    diff = y_post_pred_1[:, :n_points] - y_post_pred_0[:, :n_points]

    diff_2 = jnp.mean(y_post_pred_1, axis=1) - jnp.mean(y_post_pred_1, axis=1)
    print(diff)
    print(diff_2)
    ### Test 2
    bm = BayesModel(
        metric_type="binary", observations=ed.data.groupby("T")["conversion"]
    )
    nuts = NUTS(bm.build_model, adapt_step_size=True)
    mcmc = MCMC(nuts, num_samples=1000, num_warmup=500)
    rng_key = random.PRNGKey(0)

    bm.set_prior_model(
        "probs", dist.Uniform, {"args": (), "kwargs": {"low": 0, "high": 1}}
    )

    mcmc.run(
        rng_key,
    )
    mcmc.print_summary(exclude_deterministic=False)

    ### Test 3
    bm = BayesModel(
        metric_type="continuous", observations=ed.data.groupby("T")["revenue"]
    )
    nuts = NUTS(bm.build_model, adapt_step_size=True)
    mcmc = MCMC(nuts, num_samples=1000, num_warmup=500)
    rng_key = random.PRNGKey(0)

    bm.set_model(
        dist.Normal,
        ["loc", "scale"],
        [dist.LogNormal, dist.Gamma],
        [
            {"args": {}, "kwargs": {"loc": 0, "scale": 1}},
            {"args": (1,), "kwargs": {"rate": 0.5}},
        ],
    )

    mcmc.run(
        rng_key,
    )
    mcmc.print_summary(exclude_deterministic=False)
