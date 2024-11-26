"""Bayesian A/B testing module."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from jax import random as rd
from numpy.random import binomial, lognormal, poisson
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, init_to_feasible
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from tw_experimentation.bayes.bayes_model import BayesModel
from tw_experimentation.statistical_tests import BaseTest
from tw_experimentation.utils import ExperimentDataset, hex_to_rgb, variant_color_map

NUM_SAMPLES_MIN = 5000
NUM_SAMPLES_MAX = 15000
NUM_WARMUP = 300


@dataclass
class BayesResult:
    """Class to store the results of a Bayesian test."""

    targets: List[str]
    metric_types: List[str]
    sampler: dict[str, MCMC]
    posterior_ate: dict[str, dict[str, np.ndarray]]
    posterior_means: dict[str, dict[str, np.ndarray]]
    ate_hdi: dict[str, dict[str, List[float]]]
    means_hdi: dict[str, dict[str, List[float]]]
    hdi: float
    variant_labels: dict[int, str]
    n_variants: int
    outcome_stds: dict[str, float]
    prior_ate: Union[None, dict[str, dict[str, np.ndarray]]]
    prior_means: Union[None, dict[str, dict[str, np.ndarray]]]

    def bayes_factor(self, target: str, treatment: int):
        """
        Calculates the Bayes factor for a given target and treatment.

        Parameters:
            target (str): The target variable.
            treatment (int): The treatment index.

        Returns:
            float: The calculated Bayes factor.
        """
        assert isinstance(treatment, int) and treatment in range(1, self.n_variants)

        # Drop inf values from prior dist which strangely appear sometimes
        prior = np.array(self.prior_ate[target][treatment])
        prior = prior[~np.isinf(prior)]

        prior_pdf_smoothed = gaussian_kde(prior)
        prior_pdf_at_zero = prior_pdf_smoothed.evaluate(0)

        posterior_pdf_smoothed = gaussian_kde(self.posterior_ate[target][treatment])
        posterior_pdf_at_zero = posterior_pdf_smoothed.evaluate(0)

        bayes_factor = posterior_pdf_at_zero / prior_pdf_at_zero

        return bayes_factor[0]

    def false_discovery_rate(self, target: str, treatment: int):
        """
        Calculates the false discovery rate (FDR) for a given target and treatment.

        Parameters:
        target (str): The target variable.
        treatment (int): The treatment number.

        Returns:
        float: The false discovery rate.

        Raises:
        AssertionError: If the treatment is not an integer or is not within the
        range of 1 to the number of variants.
        """
        assert isinstance(treatment, int) and treatment in range(1, self.n_variants)

        bayes_factor = self.bayes_factor(target, treatment)

        return 1 / (1 + bayes_factor)

    def bayes_factor_decision(
        self, target: str, treatment: int, false_disc_threshold=0.01
    ):
        """
        Determines the decision based on the Bayes factor.

        Args:
            target (str): The target variable.
            treatment (int): The treatment variable.
            false_disc_threshold (float, optional): The threshold for the
                false discovery rate. Defaults to 0.01.

        Returns:
            str: The decision based on the Bayes factor. Either "accept null" or
                "reject null".
        """
        fdr = self.false_discovery_rate(target, treatment)

        if fdr > false_disc_threshold:
            message = "accept null"
        else:
            message = "reject null"
        return message

    def prob_greater_than_zero(self, target: str):
        """Compute the probability that the average treatment effect is greater than
        zero.

        Args:
            target (str): target metric
        Returns:
            dict: probabilities for each variant
        """
        return {
            k: (np.array(v) > 0).mean() for k, v in self.posterior_ate[target].items()
        }

    def prob_greater_than_z(self, z: float, target: str):
        """Compute the probability that the average treatment effect is greater than z.

        Args:
            z (float): threshold
            target (str): target metric
        Returns:
            dict: probabilities for each variant
        """
        return {
            k: (np.array(v) > z).mean() for k, v in self.posterior_ate[target].items()
        }

    def prob_smaller_than_z(self, z: float, target: str):
        """Compute the probability that the average treatment effect is smaller than z.

        Args:
            z (float): threshold
            target (str): target metric
        Returns:
            dict: probabilities for each variant
        """
        return {
            k: (np.array(v) < z).mean() for k, v in self.posterior_ate[target].items()
        }

    def prob_greater_than_z_absolute(self, z: float, target: str):
        """Compute the probability that the absolute value of the average treatment
        effect is greater than z.

        Args:
            z (float): threshold
            target (str): target metric
        Returns:
            dict: probabilities for each variant
        """
        return {
            k: (np.array(v) > np.abs(z)).mean()
            for k, v in self.posterior_ate[target].items()
        }

    def prob_within_interval(self, z_lower: float, z_upper: float, target: str):
        """Compute the probability that the average treatment effect is within the
        interval [z_lower, z_upper]

        Args:
            z_lower (float): lower bound of interval
            z_upper (float): upper bound of interval
            target (str): target metric
        Returns:
            dict: probabilities for each variant
        """
        return {
            k: ((np.array(v) > z_lower) & (np.array(v) < z_upper)).mean()
            for k, v in self.posterior_ate[target].items()
        }

    def prob_outside_interval(self, z_lower: float, z_upper: float, target: str):
        """Compute the probability that the average treatment effect is outside the
        interval [z_lower, z_upper]

        Args:
            z_lower (float): lower bound of interval
            z_upper (float): upper bound of interval
            target (str): target metric
        Returns:
            dict: probabilities for each variant
        """
        return {
            k: ((np.array(v) < z_lower) | (np.array(v) > z_upper)).mean()
            for k, v in self.posterior_ate[target].items()
        }

    def rope(
        self,
        target: str,
        rope_upper: Optional[float] = None,
        rope_lower: Optional[float] = None,
    ):
        """Compute the probability that the average treatment effect is in the region of
        practical equivalence (ROPE)

        https://easystats.github.io/bayestestR/articles/region_of_practical_equivalence.html

        Args:
            target (str): target metric
            rope_upper (float): upper bound of ROPE
            rope_lower (float): lower bound of ROPE
        Returns:
            dict: probabilities for each variant
            float: ROPE lower bound
            float: ROPE upper bound
        """
        if rope_upper is None or rope_lower is None:
            rope_lower, rope_upper = self._rope_interval_autodetect_intervals(
                target=target
            )
        return (
            {
                k: ((np.array(v) < rope_lower) | (np.array(v) > rope_upper)).mean()
                for k, v in self.posterior_ate[target].items()
            },
            rope_lower,
            rope_upper,
        )

    def _rope_interval_autodetect_intervals(
        self, target: str, scale_param: Optional[float] = 0.1
    ):
        """Compute the ROPE interval based on the standard deviation of the target
        metric.

        Args:
            target (str): target metric
            scale_param (Optional[float], optional): Cohen's d approximate .
                Defaults to .1.

        Returns:
            float, float: interval lower and upper bounds
        """
        std = self.outcome_stds[target]
        rope_upper = scale_param * std
        rope_lower = -scale_param * std
        return rope_lower, rope_upper

    def _posterior_and_hdi_plot(
        self, sample_per_variant, posterior_hdi_per_variant, distribution_opacity=0.3
    ):
        """Plot the posterior distribution and the high density interval (HDI)

        Args:
            sample_per_variant (dict): dictionary of posterior samples
            posterior_hdi_per_variant (dict): dictionary of HDI
            distribution_opacity (float): opacity of the distribution plot shades
        Returns:
            plotly figure
        """
        color_per_variant = variant_color_map(self.n_variants)
        plot_indices = [
            k - min(sample_per_variant.keys()) for k in sample_per_variant.keys()
        ]

        fig_aux1 = ff.create_distplot(
            list(sample_per_variant.values()),
            list(sample_per_variant.keys()),
            self.variant_labels,
            show_hist=False,
        )
        fig = make_subplots(
            rows=2,
            cols=1,
            row_titles=[
                "Posterior distribution",
                f"High Density Interval (HDI): {self.hdi}",
            ],
            shared_xaxes=True,
        )
        fig.update_annotations(font_size=10)

        for k in plot_indices:
            fig.add_trace(
                fig_aux1.data[k],
                row=1,
                col=1,
            )

        fig.for_each_trace(
            lambda trace: trace.update(
                marker={"color": color_per_variant[int(trace.name)]},
                name=self.variant_labels[int(trace.name)],
            )
        )

        for k, j in zip(sample_per_variant.keys(), plot_indices):
            fig.add_scatter(
                x=fig_aux1.data[j].x,
                y=fig_aux1.data[j].y,
                fill="tozeroy",
                mode="none",
                fillcolor=(
                    "rgba"
                    f"{(*hex_to_rgb(color_per_variant[k][1:]), distribution_opacity)}"
                ),
                showlegend=False,
                row=1,
                col=1,
                hoverinfo="skip",
            )
            fig.add_trace(
                go.Scatter(
                    x=list(posterior_hdi_per_variant[k].values()),
                    y=[self.variant_labels[k]] * 2,
                    mode="lines",
                    showlegend=False,
                    line={"color": color_per_variant[k], "width": 10},
                ),
                row=2,
                col=1,
            )
        return fig

    def fig_posterior_by_target(self, target: str):
        """Plot the posterior distribution and the high density interval (HDI) of the
        expected value.

        Args:
            target (str): target metric

        Returns:
            plotly figure
        """
        sample_per_variant = self.posterior_means[target]
        posterior_hdi_per_variant = self.means_hdi[target]

        fig = self._posterior_and_hdi_plot(
            sample_per_variant, posterior_hdi_per_variant
        )
        fig.update_layout(
            title_text=(
                f"{target}: Posterior distribution of the variant expected values"
            )
        )
        return fig

    def fig_posterior_cdf_by_target(self, target: str, facet_rows_variant=False):
        """Generates a plot of the empirical cumulative distribution (ECDF) function of
        treatment effect for a given target.

        Args:
            target (str): The target for which to generate the plot.
            facet_rows_variant (bool, optional): Whether to facet the plot by variant.
                Defaults to False.

        Returns:
            fig: The plotly figure object.
        """
        sample_per_variant = self.posterior_ate[target]
        posterior_hdi_per_variant = self.ate_hdi[target]

        fig = self.fig_posterior_difference_cdf(
            sample_per_variant,
            # posterior_hdi_per_variant,
            facet_rows_variant=facet_rows_variant,
        )
        fig.update_layout(
            title_text=(
                f"{target}: Empirical cumulative distribution (ECDF) function of"
                "treatment effect"
            )
        )
        return fig

    def fig_posterior_difference_by_target(self, target: str):
        """Plot the posterior distribution and the high density interval (HDI) of the
        expected treatment effect.

        Args:
            target (str): target metric
        Returns:
            plotly figure
        """
        sample_per_variant = self.posterior_ate[target]
        posterior_hdi_per_variant = self.ate_hdi[target]

        fig = self._posterior_and_hdi_plot(
            sample_per_variant, posterior_hdi_per_variant
        )
        fig.update_layout(
            title_text=(
                f"{target}: Posterior distribution of the average treatment effect"
            )
        )
        return fig

    def fig_posterior_difference_cdf(
        self,
        sample_per_variant: dict,
        facet_rows_variant: bool = False,
        shade_areas: bool = True,
        shade_limits: Tuple[Union[float, None], Union[float, None]] = (None, None),
    ) -> make_subplots:
        """Generates a plotly figure showing the cumulative density function of the
        treatment effect for each variant, based on the posterior distribution of the
        difference in means between the variant and the control group.

        Args:
            sample_per_variant (dict): A dictionary mapping variant names to lists
                of samples.
            facet_rows_variant (bool, optional): Whether to facet the plot by variant.
                Defaults to False.
            shade_areas (bool, optional): Whether to shade an area. Not implemented yet
                Defaults to True.
            shade_limits (Tuple[Union[float, None], Union[float, None]], optional):
                The lower and upper limits of the shaded area. Not implemented yet
                Defaults to (None, None).

        Returns:
            make_subplots: A plotly figure object.
        """

        # TODO: Implement shading of areas with shade_areas and shade_limits
        _ = shade_areas
        _ = shade_limits
        variant_name = "Variant"
        value_name = "Value"
        data_list = [
            {variant_name: variant, value_name: value}
            for variant, values in sample_per_variant.items()
            for value in values
        ]
        df = pd.DataFrame(data_list)

        fig_aux = px.ecdf(
            df,
            x="Value",
            color="Variant",
            title="Cumulative density function of treatment effect",
        )
        color_per_variant = variant_color_map(self.n_variants)
        fig_aux.for_each_trace(
            lambda trace: trace.update(
                marker=dict(color=color_per_variant[int(trace.name)]),
                name=self.variant_labels[int(trace.name)],
            )
        )

        if facet_rows_variant:
            rows = {k: k for k in range(1, self.n_variants)}
            n_rows = len(rows)
        else:
            rows = {k: 1 for k in range(1, self.n_variants)}
            n_rows = 1

        fig = make_subplots(rows=n_rows)
        for k, v in rows.items():
            fig.add_trace(fig_aux.data[k - 1], row=v, col=1)

        return fig


class BayesTest(BaseTest):
    """Class to perform Bayesian A/B testing."""

    def __init__(
        self,
        ed: ExperimentDataset,
    ):
        """
        Initializes a BayesianTest object.

        Args:
            ed (ExperimentDataset): The ExperimentDataset object containing the data
                for the experiment.

        Attributes:
            sampler (dict): Dictionary to store the sampler object.
            posterior_samples (dict): Dictionary to store the posterior
                samples.
            posterior_samples_difference (dict): Dictionary to store the
                posterior samples difference.
            posterior_ate (dict): Dictionary to store the posterior
                average treatment effect.
            posterior_means (dict): Dictionary to store the posterior means.
            posterior_hdi (dict): Dictionary to store the posterior
                highest density interval.
            posterior_difference_hdi (dict): Dictionary to store the
                posterior difference highest density interval.
            prior_samples (dict): Dictionary to store the prior samples.
            prior_samples_difference (dict): Dictionary to store the prior
                samples difference.
            prior_ate (dict): Dictionary to store the prior average treatment
                effect.
            prior_means (dict): Dictionary to store the prior means.
            post_pred (dict): Dictionary to store the posterior predictive samples.
            post_pred_diff (dict): Dictionary to store the posterior predictive
                samples difference.
            post_pred_mean_distribution (dict): Dictionary to store the posterior
                predictive mean distribution.
            post_pred_mean_distribution_diff (dict): Dictionary to store the
                posterior predictive mean distribution difference.
            hdi (float): The highest density interval.
            num_samples (int): The number of samples to be used in the analysis.

        """
        super().__init__(ed)

        self.sampler = {}
        self.posterior_samples = {}
        self.posterior_samples_difference = {}
        self.posterior_ate = {}
        self.posterior_means = {}
        self.posterior_hdi = {}
        self.posterior_difference_hdi = {}
        self.prior_samples = {}
        self.prior_samples_difference = {}
        self.prior_ate = {}
        self.prior_means = {}
        self.post_pred = {}
        self.post_pred_diff = {}
        self.post_pred_mean_distribution = {}
        self.post_pred_mean_distribution_diff = {}
        self.set_model_to_default()
        self.hdi = 0.95

        self.num_samples = min(
            NUM_SAMPLES_MAX, max(NUM_SAMPLES_MIN, self.ed.data.shape[0])
        )

    def set_model(
        self,
        target: str,
        likelihood_model,
        variables: List[str],
        prior_models: List,
        params_models: List[dict],
    ):
        """
        Sets the likelihood model, variables, prior models, and parameters models for a
            given target.

        Args:
            target (str): The target for which the models are being set.
            likelihood_model: The likelihood model for the target.
            variables (List[str]): The list of variables for the target.
            prior_models (List): The list of prior models for the variables.
            params_models (List[dict]): The list of parameter models for the variables.

        Returns:
            None
        """

        self.likelihood_model_per_target[target] = likelihood_model

        self.variables_per_target[target] = variables
        self.prior_models_per_target[target] = dict(zip(variables, prior_models))
        self.params_models_per_target[target] = dict(zip(variables, params_models))

        # reset prior_modle and prior_model_params
        # self.prior_model = {}
        # self.prior_model_params = {}

        # for model_and_params in zip(variables, prior_models, params_models):
        #    self.set_prior_model(*model_and_params)

    def set_prior_model(self, target, variable: str, model, model_params: dict):
        """Set prior model for a likelihood model parameter.

        Args:
            variable (str): name of variab model variable to be fed into likelihood
            model (...): numpyro distribution used as prior
            model_params (dict): args and kwargs for prior model
                                Must be of form
                                {'args': (a,b),
                                 'kwargs': {'c': d}
                                 }
        """
        if target not in self.params_models_per_target:
            self.prior_models_per_target[target] = {}

        self.prior_models_per_target[target][variable] = model
        self.update_prior_model_param(target, variable, model_params)

    def update_prior_model_param(self, target, variable: str, model_params: dict):
        """Update parameters for prior. The prior model must have been defined before.

        Args:
            variable (str):     name of variab model variable to be fed into likelihood
            model_params (dict): args and kwargs for prior model
                                Must be of form
                                {'args': (a,b),
                                 'kwargs': {'c': d}
                                 }
        """
        if target not in self.params_models_per_target:
            self.params_models_per_target[target] = {}

        self.params_models_per_target[target][variable] = model_params

    def set_model_to_default(self):
        """Reset the bayesian model to default settings."""
        self.likelihood_model_per_target = {}
        self.variables_per_target = {}
        self.prior_models_per_target = {}
        self.params_models_per_target = {}

    def _setup_bayesmodel(self, target, fit_model=True):
        """
        Set up a BayesModel object for the given target variable.

        Args:
            target (str): The name of the target variable.
            fit_model (bool, optional): Whether to perform
                Bayesian updating. Defaults to True.

        Returns:
            BayesModel: The initialized BayesModel object.

        Raises:
            None

        """
        bm = BayesModel(
            metric_type=self.ed.metric_types[target],
            observations=self.ed.data.groupby(self.ed.variant)[target],
            n_variants=self.ed.n_variants,
            do_bayesian_updating=fit_model,
        )

        if (
            target in self.likelihood_model_per_target
            and target in self.variables_per_target
        ):
            bm.set_model(
                self.likelihood_model_per_target[target],
                self.variables_per_target[target],
                list(self.prior_models_per_target[target].values()),
                list(self.params_models_per_target[target].values()),
            )
        elif target in self.params_models_per_target:
            for variable in self.params_models_per_target[target]:
                bm.set_prior_model_param(
                    variable, self.params_models_per_target[target][variable]
                )
        self.variables_per_target[target] = bm.get_likelihood_variables()
        return bm

    def compute_posterior(self, store_prior=True, compute_bayes_factor=True, verbose=0):
        """Run the Bayesian model via numpyro to obtain the posterior distribution."""

        # TODO: save priors on this level

        for target in self.ed.targets:
            bm = self._setup_bayesmodel(target, fit_model=True)
            nuts = NUTS(
                bm.build_model, adapt_step_size=True, init_strategy=init_to_feasible()
            )
            mcmc = MCMC(
                nuts,
                num_samples=self.num_samples,
                num_warmup=NUM_WARMUP,
            )
            rng_key = rd.PRNGKey(0)

            mcmc.run(
                rng_key,
            )
            self.sampler[target] = mcmc
            self._store_posterior_samples(target)
            self._store_posterior_ate(target)
            self._store_posterior_means(target)
            self._update_hdi(target, prob=self.hdi)

            if verbose == 1:
                self.sampler[target].print_summary(exclude_deterministic=False)

        # self._compute_posterior_predictive()

        if compute_bayes_factor:
            self.compute_bayes_factor()

        self.fitted = True

        return BayesResult(
            targets=self.ed.targets,
            metric_types=self.ed.metric_types,
            sampler=self.sampler,
            posterior_ate=self.posterior_ate,
            posterior_means=self.posterior_means,
            ate_hdi=self.ate_hdi,
            means_hdi=self.means_hdi,
            hdi=self.hdi,
            variant_labels=self.ed.variant_names,
            n_variants=self.ed.n_variants,
            outcome_stds=self.ed.target_standard_deviations,
            prior_ate=self.prior_ate if compute_bayes_factor else None,
            prior_means=self.prior_means if compute_bayes_factor else None,
        )

    def get_summary(self, target):
        """Summarize the Bayes model for a given target."""
        self.sampler[target].print_summary(exclude_deterministic=False)

    def compute_bayes_factor(self):
        """
        Computes the Bayes factor for each target in the experiment.

        Returns:
            None
        """
        # https://arxiv.org/abs/1602.05549
        for target in self.ed.targets:
            bm = self._setup_bayesmodel(target, fit_model=False)
            nuts = NUTS(
                bm.build_model, adapt_step_size=True, init_strategy=init_to_feasible()
            )
            N_PRIOR_SAMPLE = 2000
            mcmc = MCMC(
                nuts,
                num_samples=N_PRIOR_SAMPLE,
                num_warmup=NUM_WARMUP,
            )
            rng_key = rd.PRNGKey(0)

            mcmc.run(
                rng_key,
            )
            self._store_prior_samples(target, mcmc)
            self._store_prior_ate(target, mcmc)
            self._store_prior_means(target, mcmc)

    def _compute_posterior_predictive(self):
        """Compute posterior predictive distribution from posterior samples only
        possible after model fit."""
        n_samples_post_pred = 100000
        for target in self.ed.targets:
            self.post_pred[target] = {}
            for k in range(self.ed.n_variants):
                if self.ed.metric_types[target] == "binary":
                    self.post_pred[target][k] = binomial(
                        1,
                        np.random.choice(
                            self.posterior_samples[target]["probs"][k],
                            size=(n_samples_post_pred,),
                        ),
                    )
                elif self.ed.metric_types[target] == "discrete":
                    self.post_pred[target][k] = binomial(
                        1,
                        1
                        - np.random.choice(
                            self.posterior_samples[target]["gate"][k],
                            size=(n_samples_post_pred,),
                        ),
                    ) * poisson(
                        lam=np.random.choice(
                            self.posterior_samples[target]["rate"][k],
                            size=(n_samples_post_pred,),
                        ),
                    )
                elif self.ed.metric_types[target] == "continuous":
                    self.post_pred[target][k] = binomial(
                        1,
                        1
                        - np.random.choice(
                            self.posterior_samples[target]["gate"][k],
                            size=(n_samples_post_pred,),
                        ),
                    ) * lognormal(
                        mean=np.random.choice(
                            self.posterior_samples[target]["loc"][k],
                            size=(n_samples_post_pred,),
                        ),
                        sigma=np.random.choice(
                            self.posterior_samples[target]["scale"][k],
                            size=(n_samples_post_pred,),
                        ),
                    )

            self.post_pred_diff[target] = {}
            for k in range(1, self.ed.n_variants):
                self.post_pred_diff[k] = (
                    self.post_pred[target][k] - self.post_pred[target][0]
                )

    def compute_posterior_predictive_mean(self):
        n_samples_post_pred = 2000
        n_sample_means = 200
        for target in self.ed.targets:
            self.post_pred_mean_distribution[target] = {}
            for k in range(self.ed.n_variants):
                if self.ed.metric_types[target] == "binary":
                    self.post_pred_mean_distribution[target][k] = np.array(
                        [
                            binomial(
                                1,
                                np.random.choice(
                                    self.posterior_samples[target]["probs"][k],
                                    size=(n_samples_post_pred,),
                                ),
                            ).mean()
                            for _ in range(n_sample_means)
                        ]
                    )
                elif self.ed.metric_types[target] == "discrete":
                    self.post_pred_mean_distribution[target][k] = np.array(
                        [
                            (
                                binomial(
                                    1,
                                    1
                                    - np.random.choice(
                                        self.posterior_samples[target]["gate"][k],
                                        size=(n_samples_post_pred,),
                                    ),
                                )
                                * poisson(
                                    lam=np.random.choice(
                                        self.posterior_samples[target]["rate"][k],
                                        size=(n_samples_post_pred,),
                                    ),
                                )
                            ).mean()
                            for _ in range(n_sample_means)
                        ]
                    )
                elif self.ed.metric_types[target] == "continuous":
                    self.post_pred_mean_distribution[target][k] = np.array(
                        [
                            (
                                binomial(
                                    1,
                                    1
                                    - np.random.choice(
                                        self.posterior_samples[target]["gate"][k],
                                        size=(n_samples_post_pred,),
                                    ),
                                )
                                * lognormal(
                                    mean=np.random.choice(
                                        self.posterior_samples[target]["loc"][k],
                                        size=(n_samples_post_pred,),
                                    ),
                                    sigma=np.random.choice(
                                        self.posterior_samples[target]["scale"][k],
                                        size=(n_samples_post_pred,),
                                    ),
                                )
                            ).mean()
                            for _ in range(n_sample_means)
                        ]
                    )

            self.post_pred_mean_distribution_diff[target] = {}
            for k in range(1, self.ed.n_variants):
                self.post_pred_mean_distribution_diff[target][k] = (
                    self.post_pred_mean_distribution[target][k]
                    - self.post_pred_mean_distribution[target][0]
                )

    def compute_loss(self):
        """compute one of different loss functions"""
        raise NotImplementedError

    def compute_gain(self):
        """ompute gain function as chosen"""
        raise NotImplementedError

    def compute_utility(self):
        """compute utility function with expected gain versus loss"""
        raise NotImplementedError

    def plot_prior(self):
        raise NotImplementedError

    def plot_dynamic_gain_loss(self):
        """we need a function that create daily, weekly etc. batches here"""
        raise NotImplementedError

    def plot_dynamic_bayes_factor(self):
        """dynamic bayes factor plot"""
        raise NotImplementedError

    def get_interpretation(self):
        # 'Under the assumption that all conversion rates are equally likely,
        # there is an x% chance that the treatment effect is negative' ...
        pass

    def _store_posterior_samples(self, target):
        """
        Method to store the posterior samples for a given target.
        posterior samples from the mcmc object
        #where it is hidden a bit

        Parameters:
        - target: The target for which to store the posterior samples.

        Returns:
        None
        """
        self.posterior_samples[target] = {}
        for variable in self.variables_per_target[target]:
            self.posterior_samples[target][variable] = {}

            for k in range(self.ed.n_variants):
                self.posterior_samples[target][variable][k] = list(
                    self.sampler[target].get_samples()[f"variant_{k}_{variable}"]
                )
        self.posterior_samples_difference[target] = {}
        for variable in self.variables_per_target[target]:
            self.posterior_samples_difference[target][variable] = {}

            for k in range(1, self.ed.n_variants):
                self.posterior_samples_difference[target][variable][k] = list(
                    self.sampler[target].get_samples()[f"delta_{k}_{variable}"]
                )

    def _store_posterior_means(self, target):
        """
        Stores the posterior means for each variant of the target variable.

        Parameters:
        target (str): The name of the target variable.

        Returns:
        None
        """
        self.posterior_means[target] = {}
        for k in range(self.ed.n_variants):
            self.posterior_means[target][k] = list(
                self.sampler[target].get_samples()[f"means_{k}"]
            )

    def _store_posterior_ate(self, target):
        """
        Stores the posterior average treatment effect (ATE) for a given target variable.

        Parameters:
            target (str): The target variable for which to store the posterior ATE.

        Returns:
            None
        """
        self.posterior_ate[target] = {}

        for k in range(1, self.ed.n_variants):
            self.posterior_ate[target][k] = list(
                self.sampler[target].get_samples()[f"ate_{k}"]
            )

    def _store_prior_samples(self, target, prior_sampler):
        """
        Method to store the prior samples for a given target variable.

        Parameters:
        - target (str): The target variable for which the prior samples are
            stored.
        - prior_sampler (object): The prior sampler object that contains
            the prior samples.

        Returns:
        None
        """
        self.prior_samples[target] = {}
        for variable in self.variables_per_target[target]:
            self.prior_samples[target][variable] = {}

            for k in range(self.ed.n_variants):
                self.prior_samples[target][variable][k] = list(
                    prior_sampler.get_samples()[f"variant_{k}_{variable}"]
                )
        self.prior_samples_difference[target] = {}
        for variable in self.variables_per_target[target]:
            self.prior_samples_difference[target][variable] = {}

            for k in range(1, self.ed.n_variants):
                self.prior_samples_difference[target][variable][k] = list(
                    prior_sampler.get_samples()[f"delta_{k}_{variable}"]
                )

    def _store_prior_means(self, target, prior_sampler):
        """
        Stores the prior means for each variant of the target variable.

        Parameters:
        target (str): The name of the target variable.
        prior_sampler (PriorSampler): The prior sampler object.

        Returns:
        None
        """
        self.prior_means[target] = {}
        for k in range(self.ed.n_variants):
            self.prior_means[target][k] = list(
                prior_sampler.get_samples()[f"means_{k}"]
            )

    def _store_prior_ate(self, target, prior_sampler):
        """
        Stores the prior average treatment effect (ATE) for a given target and variant.

        Args:
            target (str): The target variable.
            prior_sampler (PriorSampler): The prior sampler object.

        Returns:
            None
        """
        self.prior_ate[target] = {}

        for k in range(1, self.ed.n_variants):
            self.prior_ate[target][k] = list(prior_sampler.get_samples()[f"ate_{k}"])

    @property
    def ate_hdi(self, prob=0.95):
        """
        Calculate the highest density interval (HDI) for the average treatment
            effect (ATE) for each target and variant.

        Parameters:
            prob (float): The probability mass to be included in the HDI.
                Default is 0.95.

        Returns:
            dict: A dictionary containing the HDI for each target and variant.
                The structure of the dictionary is as follows:
                {
                    target1: {
                        variant1: {
                            "min": minimum_value,
                            "max": maximum_value
                        },
                        variant2: {
                            "min": minimum_value,
                            "max": maximum_value
                        },
                        ...
                    },
                    target2: {
                        ...
                    },
                    ...
                }
        """
        self.hdi = 0.95
        ate_hdi = {}
        for target in self.ed.targets:
            ate_hdi[target] = {}
            for k in range(1, self.ed.n_variants):
                ate_hdi[target][k] = dict(
                    zip(
                        ["min", "max"],
                        hpdi(
                            self.sampler[target].get_samples()[f"ate_{k}"],
                            prob=prob,
                        ),
                    )
                )
        return ate_hdi

    @property
    def means_hdi(self, prob=0.95):
        """
        Calculate the highest posterior density interval (HDI)
            for the means of each target and variant.

        Parameters:
        prob (float): The probability of the HDI. Default is 0.95.

        Returns:
        dict: A dictionary containing the HDI for each target and variant.
        """
        self.hdi = 0.95
        means_hdi = {}
        for target in self.ed.targets:
            means_hdi[target] = {}
            for k in range(self.ed.n_variants):
                means_hdi[target][k] = dict(
                    zip(
                        ["min", "max"],
                        hpdi(
                            self.sampler[target].get_samples()[f"means_{k}"],
                            prob=prob,
                        ),
                    )
                )
        return means_hdi

    def _update_hdi(self, target, prob=0.95):
        """
        Update the highest density interval (HDI) for the given target variable.

        Parameters:
            target (str): The target variable for which to update the HDI.
            prob (float, optional): The probability mass to include in the HDI.
                Defaults to 0.95.
        """
        self.hdi = prob
        self.posterior_hdi[target] = {}
        for variable in self.variables_per_target[target]:
            self.posterior_hdi[target][variable] = {}

            for k in range(self.ed.n_variants):
                self.posterior_hdi[target][variable][k] = dict(
                    zip(
                        ["min", "max"],
                        hpdi(
                            self.sampler[target].get_samples()[
                                f"variant_{k}_{variable}"
                            ],
                            prob=prob,
                        ),
                    )
                )

        self.posterior_difference_hdi[target] = {}
        for variable in self.variables_per_target[target]:
            self.posterior_difference_hdi[target][variable] = {}

            for k in range(1, self.ed.n_variants):
                self.posterior_difference_hdi[target][variable][k] = dict(
                    zip(
                        ["min", "max"],
                        hpdi(
                            self.sampler[target].get_samples()[f"delta_{k}_{variable}"],
                            prob=prob,
                        ),
                    )
                )

    def set_hdi(self, prob=0.95):
        """
        Sets the highest density interval (HDI) for the given probability.

        Parameters:
            prob (float): The probability for the HDI. Default is 0.95.

        Returns:
            None
        """
        self.hdi = prob
        for target in self.ed.targets:
            self._update_hdi(target, prob)
