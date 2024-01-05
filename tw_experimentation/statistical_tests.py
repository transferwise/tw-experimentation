from tw_experimentation.utils import ExperimentDataset

from abc import ABC, abstractmethod

from tw_experimentation.variance_reduction.cuped import (
    CUPED,
    MultivariateCUPED,
)

from dataclasses import dataclass
from typing import Optional, Dict, Union, List

import numpy as np
import pandas as pd
import scipy.stats as sps


# TODO: Write factory so that the frequentist test result can be calculated for each segment?
@dataclass
class FrequentistPerTargetResults:
    variant_means: Dict[int, float]
    variant_stds: Dict[int, float]
    variant_n_samples: Dict[int, int]

    def __post_init__(self):
        self.n_treatments = len(self.variant_means.keys()) - 1

    def t_test_per_variant(self, direction="two-sided", pooled_variance: bool = True):
        p_values = dict()
        t_stats = dict()
        for variant in self.variant_means.keys():
            if variant > 0:
                t_stat, p_value = self.t_test(
                    variant, direction=direction, pooled_variance=pooled_variance
                )
                p_values[variant] = p_value
                t_stats[variant] = t_stat

        return p_values, t_stats

    def t_test_conf_interval_per_variant(
        self,
        direction="two-sided",
        pooled_variance: bool = True,
        alphas: Union[float, List[float]] = 0.05,
    ):
        if isinstance(alphas, float):
            alphas = [alphas] * len(self.variant_means.keys())
        else:
            assert isinstance(alphas, list) and len(alphas) == len(
                self.variant_means.keys()
            ), "alphas must be a list of length equal to the number of variants"

        cis_lower = dict()
        cis_upper = dict()
        for index, variant in enumerate(self.variant_means.keys()):
            if variant > 0:
                ci_lower, ci_upper = self.t_test_confidence_interval(
                    variant,
                    alpha=alphas[index],
                    direction=direction,
                    pooled_variance=pooled_variance,
                )
                cis_lower[variant] = ci_lower
                cis_upper[variant] = ci_upper
        return cis_lower, cis_upper

    def t_test(
        self, treatment, direction: str = "two-sided", pooled_variance: bool = True
    ):
        mean_treat = self.variant_means[treatment]
        mean_control = self.variant_means[0]
        std_treat = self.variant_stds[treatment]
        std_control = self.variant_stds[0]
        n_treat = self.variant_n_samples[treatment]
        n_control = self.variant_n_samples[0]

        se = self.t_test_standard_error(
            std_treat, std_control, n_treat, n_control, pooled_variance
        )

        t_stat = (mean_treat - mean_control) / se

        if direction == "two-sided":
            p_value = 2 * sps.t.cdf(-np.abs(t_stat), n_treat + n_control - 2)
        elif direction == "greater":
            p_value = sps.t.cdf(-np.abs(t_stat), n_treat + n_control - 2)
        elif direction == "less":
            p_value = sps.t.cdf(t_stat, n_treat + n_control - 2)
        return t_stat, p_value

    @staticmethod
    def t_test_standard_error(
        std_treat, std_control, n_treat, n_control, pooled_variance: bool = True
    ):
        if pooled_variance:
            var_pooled = (
                (n_treat - 1) * std_treat**2 + (n_control - 1) * std_control**2
            ) / (n_treat + n_control - 2)
            se = np.sqrt(var_pooled * (1 / n_treat + 1 / n_control))

        else:
            se = np.sqrt(std_treat**2 / n_treat + std_control**2 / n_control)
        return se

    def t_test_confidence_interval(
        self,
        treatment,
        direction: str = "two-sided",
        alpha: float = 0.05,
        pooled_variance: bool = True,
    ):
        mean_treat = self.variant_means[treatment]
        mean_control = self.variant_means[0]
        std_treat = self.variant_stds[treatment]
        std_control = self.variant_stds[0]
        n_treat = self.variant_n_samples[treatment]
        n_control = self.variant_n_samples[0]
        se = self.t_test_standard_error(
            std_treat, std_control, n_treat, n_control, pooled_variance
        )
        if direction == "two-sided":
            critical_value = sps.t.ppf(1 - alpha / 2, n_treat + n_control - 2)
            ci_lower = (mean_treat - mean_control) - critical_value * se
            ci_upper = (mean_treat - mean_control) + critical_value * se
        elif direction == "greater":
            ci_upper = np.inf
            critical_value = sps.t.ppf(1 - alpha, n_treat + n_control - 2)
            ci_lower = -critical_value * se + (mean_treat - mean_control)
        elif direction == "less":
            ci_lower = -np.inf
            critical_value = sps.t.ppf(alpha, n_treat + n_control - 2)
            ci_upper = critical_value * se + (mean_treat - mean_control)

        return ci_lower, ci_upper

    def relative_naive_treatment_effect(self, treatment):
        try:
            relative_te = (
                (self.variant_means[treatment] - self.variant_means[0])
                / self.variant_means[0]
                * 100
            )
        except ZeroDivisionError:
            relative_te = 0
        return relative_te

    def relative_naive_treatment_effect_per_variant(self):
        relative_tes = dict()
        for variant in self.variant_means.keys():
            if variant > 0:
                relative_tes[variant] = self.relative_naive_treatment_effect(variant)
        return relative_tes

    def absolute_naive_treatment_effect(self, treatment):
        return self.variant_means[treatment] - self.variant_means[0]

    def absolute_naive_treatment_effect_per_variant(self):
        absolute_tes = dict()
        for variant in self.variant_means.keys():
            if variant > 0:
                absolute_tes[variant] = self.absolute_naive_treatment_effect(variant)
        return absolute_tes

    def two_sample_proportion_test(
        self, treatment, direction="two-sided", pooled_variance: bool = True
    ):
        mean_treat = self.variant_means[treatment]
        mean_control = self.variant_means[0]
        n_treat = self.variant_n_samples[treatment]
        n_control = self.variant_n_samples[0]

        se = self.two_sample_proportion_standard_error(
            mean_treat, mean_control, n_treat, n_control, pooled_variance
        )
        z_stat = (mean_treat - mean_control) / se
        if direction == "two-sided":
            p_value = 2 * sps.norm.cdf(-np.abs(z_stat))
        elif direction == "greater":
            p_value = sps.norm.cdf(-np.abs(z_stat))
        elif direction == "less":
            p_value = sps.norm.cdf(z_stat)

        return z_stat, p_value

    @staticmethod
    def two_sample_proportion_standard_error(
        mean_treat, mean_control, n_treat, n_control, pooled_variance: bool = True
    ):
        sample_proportion = (mean_treat * n_treat + mean_control * n_control) / (
            n_treat + n_control
        )
        if pooled_variance:
            se = np.sqrt(
                sample_proportion
                * (1 - sample_proportion)
                * (1 / n_treat + 1 / n_control)
            )
        else:
            se = np.sqrt(
                mean_treat * (1 - mean_treat) / n_treat
                + mean_control * (1 - mean_control) / n_control
            )
        return se

    def two_sample_proportion_confidence_interval(
        self,
        treatment,
        alpha: float = 0.05,
        direction: str = "two-sided",
        pooled_variance: bool = True,
    ):
        mean_treat = self.variant_means[treatment]
        mean_control = self.variant_means[0]
        n_treat = self.variant_n_samples[treatment]
        n_control = self.variant_n_samples[0]
        se = self.two_sample_proportion_standard_error(
            mean_treat, mean_control, n_treat, n_control, pooled_variance
        )
        if direction == "two-sided":
            critical_value = sps.norm.ppf(1 - alpha / 2)
            ci_lower = (mean_treat - mean_control) - critical_value * se
            ci_upper = (mean_treat - mean_control) + critical_value * se
        elif direction == "greater":
            ci_upper = np.inf
            critical_value = sps.norm.ppf(1 - alpha)
            ci_lower = -critical_value * se + (mean_treat - mean_control)
        elif direction == "less":
            ci_lower = -np.inf
            critical_value = sps.norm.ppf(alpha)
            ci_upper = critical_value * se + (mean_treat - mean_control)

        return ci_lower, ci_upper

    def two_sample_proportion_confidence_interval_per_variant(
        self,
        direction="two-sided",
        pooled_variance: bool = True,
        alphas: Union[float, List[float]] = 0.05,
    ):
        if isinstance(alphas, float):
            alphas = [alphas] * len(self.variant_means.keys())
        else:
            assert isinstance(alphas, list) and len(alphas) == len(
                self.variant_means.keys()
            ), "alphas must be a list of length equal to the number of variants"
        cis_lower = dict()
        cis_upper = dict()
        for index, variant in enumerate(self.variant_means.keys()):
            if variant > 0:
                ci_lower, ci_upper = self.two_sample_proportion_confidence_interval(
                    variant,
                    alpha=alphas[index],
                    direction=direction,
                    pooled_variance=pooled_variance,
                )
                cis_lower[variant] = ci_lower
                cis_upper[variant] = ci_upper
        return cis_lower, cis_upper

    def two_sample_proportion_test_per_variant(
        self, direction="two-sided", pooled_variance: bool = True
    ):
        p_values = dict()
        z_stats = dict()
        for variant in self.variant_means.keys():
            if variant > 0:
                z_stat, p_value = self.two_sample_proportion_test(
                    variant, direction=direction, pooled_variance=pooled_variance
                )
                p_values[variant] = p_value
                z_stats[variant] = z_stat

        return p_values, z_stats


@dataclass
class FrequentistTestResults:
    target_variant_means: Dict[str, Dict[str, float]]
    target_variant_stds: Dict[str, Dict[str, float]]
    variant_n_samples: Dict[str, Dict[str, int]]
    target_metric_types: Dict[str, str]
    n_variants: int

    def __post_init__(self):
        self.n_targets = len(self.target_variant_means.keys())
        self.n_treatments = self.n_variants - 1
        self.stats_per_target = None
        self.CONTROL_LABEL = 0

    def compute_stats_per_target(
        self,
        direction: str = "two-sided",
        pooled_variance: bool = False,
        type_i_error: float = 0.05,
        multitest_correction: Optional[str] = None,
    ):
        """
        Compute statistical tests and confidence intervals for each target variable.

        Args:
            direction (str, optional): The direction of the test. Defaults to "two-sided".
                Options are ['two-sided', 'greater', 'less'].
            pooled_variance (bool, optional): Whether to use pooled variance for hypothesis test. Defaults to True.
            type_i_error (float, optional): The significance level(s) for confidence intervals. Defaults to 0.05.
            multitest_correction (str, optional): The method for multiple hypothesis correction. Defaults to None.
                Can be one of ['bonferroni', None].

        Returns:
            dict: A dictionary containing the computed statistical tests, p-values, confidence intervals, and treatment effects for each target variable.
        """
        # TODO: initiate multiple hypothesis correction here
        if multitest_correction is None:
            mhc = NoCorrection(type_i_error=type_i_error)
        elif multitest_correction == "bonferroni":
            mhc = BonferroniCorrection(
                type_i_error=type_i_error, n_treatments=self.n_treatments
            )
        else:
            raise NotImplementedError
        stats_per_target = dict()
        for target in self.target_variant_means.keys():
            fptr = FrequentistPerTargetResults(
                variant_means=self.target_variant_means[target],
                variant_stds=self.target_variant_stds[target],
                variant_n_samples=self.variant_n_samples,
            )
            relative_tes = fptr.relative_naive_treatment_effect_per_variant()
            absolute_tes = fptr.absolute_naive_treatment_effect_per_variant()
            if self.target_metric_types[target] in ["continuous", "discrete"]:
                p_values, t_stats = fptr.t_test_per_variant(
                    direction=direction, pooled_variance=pooled_variance
                )
                stats_per_target[target] = {
                    "test_stats": t_stats,
                    "p_values": p_values,
                    "test_type": "t_test",
                }
            elif self.target_metric_types[target] == "binary":
                p_values, z_stats = fptr.two_sample_proportion_test_per_variant(
                    direction=direction, pooled_variance=pooled_variance
                )
                stats_per_target[target] = {
                    "test_stats": z_stats,
                    "p_values": p_values,
                    "test_type": "proportion_test",
                }
            alphas = mhc.correct_alphas(p_values=p_values)
            assert isinstance(
                alphas, float
            ), "alphas must be a float, alternative not implemented yet"
            stats_per_target[target]["p_values"] = mhc.correct_p_values(
                p_values=p_values
            )
            stats_per_target[target]["absolute_tes"] = absolute_tes
            stats_per_target[target]["relative_tes"] = relative_tes

            if self.target_metric_types[target] in ["continuous", "discrete"]:
                cis_lower, cis_upper = fptr.t_test_conf_interval_per_variant(
                    direction=direction,
                    pooled_variance=pooled_variance,
                    alphas=alphas,
                )
                stats_per_target[target]["cis_lower"] = cis_lower
                stats_per_target[target]["cis_upper"] = cis_upper
            elif self.target_metric_types[target] == "binary":
                (
                    cis_lower,
                    cis_upper,
                ) = fptr.two_sample_proportion_confidence_interval_per_variant(
                    direction=direction,
                    pooled_variance=pooled_variance,
                    alphas=alphas,
                )
                stats_per_target[target]["cis_lower"] = cis_lower
                stats_per_target[target]["cis_upper"] = cis_upper
            else:
                raise NotImplementedError

            stats_per_target[target]["is_significant"] = {
                k: v < alphas for k, v in p_values.items()
            }

        self.stats_per_target = stats_per_target
        return self

    def get_results_table(self):
        assert self.stats_per_target is not None, "compute_stats_per_target first"
        targets = self.target_variant_means.keys()
        results = dict()
        for variant in range(1, self.n_variants):
            df_partial_results = pd.DataFrame(
                {
                    "Control_Group_Count": [
                        self.variant_n_samples.get(self.CONTROL_LABEL, 0)
                    ]
                    * self.n_targets,
                    "Treatment_Group_Count": [self.variant_n_samples.get(variant, 0)]
                    * self.n_targets,
                    "Control_Group_Mean": [
                        self.target_variant_means[target].get(
                            self.CONTROL_LABEL, np.nan
                        )
                        for target in targets
                    ],
                    "Treatment_Group_Mean": [
                        self.target_variant_means[target].get(variant, np.nan)
                        for target in targets
                    ],
                    "Estimated_Effect_absolute": [
                        self.stats_per_target[target]["absolute_tes"].get(
                            variant, np.nan
                        )
                        for target in targets
                    ],
                    "Estimated_Effect_relative": [
                        self.stats_per_target[target]["relative_tes"].get(
                            variant, np.nan
                        )
                        for target in targets
                    ],
                    "p_value": [
                        self.stats_per_target[target]["p_values"].get(variant, 1)
                        for target in targets
                    ],
                    "is_significant": [
                        self.stats_per_target[target]["is_significant"].get(
                            variant, False
                        )
                        for target in targets
                    ],
                    "test_statistic": [
                        self.stats_per_target[target]["test_stats"].get(variant, np.nan)
                        for target in targets
                    ],
                    "Test_type": [
                        self.stats_per_target[target]["test_type"] for target in targets
                    ],
                    "CI_lower": [
                        self.stats_per_target[target]["cis_lower"].get(variant, -np.inf)
                        for target in targets
                    ],
                    "CI_upper": [
                        self.stats_per_target[target]["cis_upper"].get(variant, np.inf)
                        for target in targets
                    ],
                },
                targets,
            )
            results[variant] = df_partial_results
        results_table = pd.concat(results.values(), axis=0, keys=results.keys())
        results_table.index.names = ["Variant", "Outcome"]
        results_table["p_value"] = results_table["p_value"].round(4)
        results_table["Estimated_Effect_relative"] = results_table[
            "Estimated_Effect_relative"
        ].replace(np.inf, 0)
        return results_table


def compute_frequentist_results(
    ed: ExperimentDataset,
):
    variant_n_samples = ed.data[ed.variant].value_counts().to_dict()

    target_variant_means = ed.data.groupby(ed.variant)[ed.targets].mean().to_dict()
    target_variant_stds = ed.data.groupby(ed.variant)[ed.targets].std().to_dict()
    target_metric_types = {k: v for k, v in ed.metric_types.items() if k in ed.targets}
    n_variants = ed.n_variants
    ftr = FrequentistTestResults(
        target_variant_means=target_variant_means,
        target_variant_stds=target_variant_stds,
        variant_n_samples=variant_n_samples,
        target_metric_types=target_metric_types,
        n_variants=n_variants,
    )
    return ftr


class MulitpleHypothesisCorrection(ABC):
    def __init__(
        self,
        correct_n_variants: bool = True,
        correct_n_targets: bool = False,
        correct_n_segments: bool = False,
        n_treatments: int = 1,
        n_targets: int = 1,
        n_segments: int = 0,
        type_i_error: float = 0.05,
    ):
        self.alpha = type_i_error
        self.correct_n_variants = correct_n_variants
        self.correct_n_targets = correct_n_targets
        self.correct_n_segments = correct_n_segments
        self.segments_label = None
        self.n_treatments = n_treatments
        self.n_targets = n_targets
        self.n_segments = n_segments
        self.type_i_error = type_i_error

    @abstractmethod
    def correct_p_values(self, p_values):
        pass

    @abstractmethod
    def correct_alphas(self, p_values=None):
        pass


class NoCorrection(MulitpleHypothesisCorrection):
    def correct_p_values(self, p_values: Dict[str, float]):
        return p_values

    def correct_alphas(self, p_values=None):
        return self.type_i_error


class BonferroniCorrection(MulitpleHypothesisCorrection):
    def correct_p_values(self, p_values: Dict[str, float]):
        if self.correct_n_targets or self.correct_n_segments:
            raise NotImplementedError(
                "Correction for targets and segments not implemented yet"
            )
        elif self.correct_n_variants:
            n_hypotheses = self.n_treatments
            p_values_corrected = {
                k: min(v * n_hypotheses, 1) for k, v in p_values.items()
            }
            return p_values_corrected
        else:
            return p_values

    def correct_alphas(self, p_values: Dict[str, float] = None):
        """
        Corrects the alpha values for multiple hypothesis testing.

        Args:
            p_values (list, optional): List of p-values for each hypothesis. Defaults to None.

        Raises:
            NotImplementedError: If correction for targets and segments is not implemented yet.

        Returns:
            Union[Dict, float]: corrected alpha values for each hypothesis or a single value for all
        """
        if self.correct_n_targets or self.correct_n_segments:
            raise NotImplementedError(
                "Correction for targets and segments not implemented yet"
            )
        elif self.correct_n_variants:
            alpha_corrected = self.type_i_error / self.n_treatments
            return alpha_corrected
        else:
            return self.type_i_error


class BaseTest(ABC):
    def __init__(
        self,
        ed: ExperimentDataset,
    ):
        self.ed = ed
        # decision in ['Reject', 'Accept', 'Continue', None]
        self.decision = None
        self.fitted = False

        self._naive_estimates()

    def get_decision(self):
        pass

    def _check_fitted(self):
        if not self.fitted:
            raise Exception("Model needs to be fitted first.")
        else:
            pass

    def _naive_estimates(self):
        self.naive_counts = {target: {} for target in self.ed.targets}
        for target in self.ed.targets:
            self.naive_counts[target] = {
                label: self.outcomes_of_variant(label, target).count()
                for label in self.ed.variant_labels
            }

        self.naive_means = {target: {} for target in self.ed.targets}
        for target in self.ed.targets:
            self.naive_means[target] = {
                label: self.outcomes_of_variant(label, target).mean()
                for label in self.ed.variant_labels
            }

        self.naive_treatment_effects = {target: {} for target in self.ed.targets}

        for target in self.ed.targets:
            self.naive_treatment_effects[target] = {
                label: (
                    self.naive_means[target][label]
                    - self.naive_means[target][self.ed.control_label]
                )
                for label in set(self.ed.variant_labels) - set([self.ed.control_label])
            }

        self.naive_relative_treatment_effects = {
            target: {} for target in self.ed.targets
        }

        for target in self.ed.targets:
            self.naive_relative_treatment_effects[target] = {
                label: self._naive_relative_treatment_effects(target, label)
                for label in set(self.ed.variant_labels) - set([self.ed.control_label])
            }

    def _naive_relative_treatment_effects(self, target, label):
        try:
            relative_te = (
                self.naive_treatment_effects[target][label]
                / self.naive_means[target][self.ed.control_label]
                * 100
            )
        except ZeroDivisionError:
            relative_te = 0

        return relative_te

    def outcomes_of_variant(self, variant, target):
        # TODO: also define this function in ExperimentDataset class
        return self.ed.data.loc[(self.ed.data[self.ed.variant] == variant), target]


# TODO: Will move to frequentist.py
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
)
from statsmodels.stats.proportion import (
    test_proportions_2indep as ztest_proportions_2indep,
)
from scipy import stats as sps

import copy


def t_test(sample_1, sample_2, alpha, alternative="two-sided"):
    stat, p_value, deg_freedom = ttest_ind(
        sample_1, sample_2, usevar="unequal", alternative=alternative
    )
    x_delta = sample_2.mean() - sample_1.mean()
    sd = np.sqrt(
        (sample_2.std() / np.sqrt(len(sample_2))) ** 2
        + (sample_1.std() / np.sqrt(len(sample_1))) ** 2
    )

    if alternative == "two-sided":
        conf_interval = (
            x_delta - sps.t.ppf(1 - (alpha / 2), deg_freedom) * sd,
            x_delta + sps.t.ppf(1 - (alpha / 2), deg_freedom) * sd,
        )
    return p_value, conf_interval, stat


def proportion_test(sample_1, sample_2, alpha, alternative="two-sided"):
    stat, p_value = ztest_proportions_2indep(
        sample_1.sum(),
        len(sample_1),
        sample_2.sum(),
        len(sample_2),
        alternative=alternative,
    )
    conf_interval = confint_proportions_2indep(
        sample_1.sum(),
        len(sample_1),
        sample_2.sum(),
        len(sample_2),
        alpha=alpha,
    )
    return p_value, conf_interval, stat


AVAILABLE_TESTS = {
    "binary": ["proportion_test"],
    "discrete": ["t_test"],
    "continuous": ["t_test"],
}


class FrequentistTest(BaseTest):
    """
    Frequentist static testing
    Should include
    test stat, confidence interval, p values, power,
    """

    def __init__(
        self,
        ed: ExperimentDataset,
        alpha: float = 0.05,
        beta: float = 0.8,
        multitest_correction="bonferroni",
    ):
        super().__init__(ed)

        # TODO: assert that keys agree with metric types
        self.test_types = {
            "binary": "proportion_test",
            "discrete": "t_test",
            "continuous": "t_test",
        }

        self.alpha = alpha
        self.beta = beta
        # TODO: multitest_correction in [None, 'bonferroni', ...]
        self.multitest_correction = multitest_correction

    def compute(self):
        # main method to generate results

        alpha = self.alpha
        if self.multitest_correction is not None:
            self._alpha_correction()
            alpha = self.alpha_corrected

        variant_dict = {
            variant: {}
            for variant in set(self.ed.variant_labels) - set([self.ed.control_label])
        }

        self.p_values = copy.deepcopy(variant_dict)
        self.conf_intervals = copy.deepcopy(variant_dict)
        self.test_stats = copy.deepcopy(variant_dict)

        for variant in set(self.ed.variant_labels) - set([self.ed.control_label]):
            for target in self.ed.targets:
                assert (
                    self.test_types[self.ed.metric_types[target]]
                    in AVAILABLE_TESTS[self.ed.metric_types[target]]
                )

                (
                    self.p_values[variant][target],
                    self.conf_intervals[variant][target],
                    self.test_stats[variant][target],
                ) = self._run_test(
                    self.test_types[self.ed.metric_types[target]],
                    self.outcomes_of_variant(self.ed.control_label, target),
                    self.outcomes_of_variant(variant, target),
                    alpha=alpha,
                )
            self._correct_pvalues(variant)

    @staticmethod
    def _run_test(test_type, data_control, data_treat, alpha):
        if test_type == "t_test":
            p_value, conf_interval, stat = t_test(
                data_control, data_treat, alpha, alternative="two-sided"
            )
        elif test_type == "proportion_test":
            # There seems to be a statsmodels bug forcing us to swap control and treat data here
            p_value, conf_interval, stat = proportion_test(
                data_treat, data_control, alpha, alternative="two-sided"
            )
        return p_value, conf_interval, stat

    def set_hypothesis_test(self, metric_type: str, hypothesis_test: str):
        # set test for a given metric,
        # e.g. set_hypothesis_test('continuous', 't_test')
        self.test_type[metric_type] = hypothesis_test

    def set_alpha(self):
        self.fitted = False
        pass

    def set_beta(self):
        self.fitted = False
        pass

    def set_data_model(self):
        # set new ExperimentDataset
        self.fitted = False
        pass

    def _compute_confidence_intervals(self, method: str = "analytical_or_bootstrap"):
        # method must be in ['analytical_or_bootstrap', 'no_bootstrap', 'always_bootstrap']
        pass

    def _bootstrap_confidence_intervals(self):
        # implement scipy.stats.bootstrap
        pass

    def _alpha_correction(self, correction_level="per_treatment"):
        # correction_level in ['per_treatment', 'global']
        # If global, correction parameter is n_treatments*n_targets. Else it is n_targets
        if self.multitest_correction == "bonferroni":
            n_hypotheses = (self.ed.n_variants - 1) * (
                correction_level == "per_treatment"
            ) + len(self.ed.targets) * (self.ed.n_variants - 1) * (
                correction_level == "global"
            )
            self.alpha_corrected = self.alpha / n_hypotheses
        else:
            pass

    def _correct_pvalues(self, variant, correction_level="per_treatment"):
        if self.multitest_correction == "bonferroni":
            n_hypotheses = (self.ed.n_variants - 1) * (
                correction_level == "per_treatment"
            ) + len(self.ed.targets) * (self.ed.n_variants - 1) * (
                correction_level == "global"
            )
            self.p_values[variant] = {
                target: min(self.p_values[variant][target] * n_hypotheses, 1)
                for target in self.ed.targets
            }

    def get_results_table(self):
        # control group, treatment group sizes
        # Then: one table per treatment:
        #   one row per metric:
        #       variant,
        #       control group count
        #       treatment group count
        #       control group mean (add required sample size later)
        #       treatment group mean (add required sample size later)
        #       Absolute difference
        #       Relative difference
        #       confidence interval
        #       p value
        #       is significant or not
        #       [test stat, test type] add in verbose option

        # self._check_fitted()

        results = {}
        for variant in range(1, self.ed.n_variants):
            df_result = pd.DataFrame(
                {
                    "Control_Group_Count": [
                        self.naive_counts[target][self.ed.control_label]
                        for target in self.ed.targets
                    ],
                    "Treatment_Group_Count": [
                        self.naive_counts[target][variant] for target in self.ed.targets
                    ],
                    "Control_Group_Mean": [
                        self.naive_means[target][self.ed.control_label]
                        for target in self.ed.targets
                    ],
                    "Treatment_Group_Mean": [
                        self.naive_means[target][variant] for target in self.ed.targets
                    ],
                    "Estimated_Effect_absolute": [
                        self.naive_treatment_effects[target][variant]
                        for target in self.ed.targets
                    ],
                    "Estimated_Effect_relative": [
                        self.naive_relative_treatment_effects[target][variant]
                        for target in self.ed.targets
                    ],
                    "Confidence_Interval": self.conf_intervals[variant].values(),
                    "p_value": self.p_values[variant].values(),
                    "is_significant": [
                        1 if self.p_values[variant][target] < self.alpha else 0
                        for target in self.ed.targets
                    ],
                    "test_statistic": self.test_stats[variant].values(),
                    "Test_type": [
                        self.test_types[self.ed.metric_types[target]]
                        for target in self.ed.targets
                    ],
                },
                self.ed.targets,
            )
            df_result[["CI_lower", "CI_upper"]] = pd.DataFrame(
                df_result["Confidence_Interval"].tolist(), index=df_result.index
            )
            del df_result["Confidence_Interval"]

            results[variant] = df_result
            results_table = pd.concat(results.values(), axis=0, keys=results.keys())
            results_table.index.names = ["Variant", "Outcome"]
        results_table["p_value"] = results_table["p_value"].round(4)
        results_table["Estimated_Effect_relative"] = results_table[
            "Estimated_Effect_relative"
        ].replace(np.inf, 0)
        return results_table


def cuped(ed: ExperimentDataset, has_correction: bool, alpha: float):
    """
    Applies the CUPED method to estimate treatment effects in experiment.
    Serves as a wrapper to variance reduction methods

    Args:
        ExperimentDataset (ExperimentDataset): An ExperimentDataset object containing the data to be analyzed.
        has_correction (bool): A boolean indicating whether to apply a multitest correction.
            If True, the Bonferroni correction is applied.
        alpha (float): The significance level for hypothesis testing.

    Returns:
        final_df (DataFrame): A pandas DataFrame containing the estimated treatment effects, confidence intervals,
            p-values, and significance levels for each target and treatment combination.
    """
    # check if multiple covariates are to be used
    multivariate = (
        isinstance(ed.pre_experiment_cols, list) and len(ed.pre_experiment_cols) > 1
    )

    # create dummy variables for variants
    dummified_data = pd.get_dummies(ed.data, columns=[ed.variant], drop_first=True)

    # create list of treatments
    treatments = [i for i in range(1, ed.n_variants)]

    # multitest correction NOTE: copied straight from FrequentistTest._alpha_correction
    # TODO: remove duplication, integrate with MultipleHypothesisCorrection
    multitest_correction = (
        "bonferroni" if ed.n_variants > 2 and has_correction == "Yes" else None
    )
    if multitest_correction is not None:
        if multitest_correction == "bonferroni":
            correction_level = "per_treatment"  # NOTE: hardcoded for the time being
            n_hypotheses = (ed.n_variants - 1) * (
                correction_level == "per_treatment"
            ) + len(ed.targets) * (ed.n_variants - 1) * (correction_level == "global")
            alpha_corrected = alpha / n_hypotheses
            alpha = alpha_corrected
        else:
            raise NotImplementedError
    else:
        pass

    # initiate estimator
    if multivariate:
        estimator = MultivariateCUPED()

    else:
        estimator = CUPED()

    result_dfs = []

    # run estimator for each target and treatment
    for target in ed.targets:
        for treatment in treatments:
            estimator = estimator.fit(
                data=dummified_data,
                treatment_column=ed.variant + "_" + str(treatment),
                target_column=target,
                covariate_column=ed.pre_experiment_cols[0],
                covariate_columns=ed.pre_experiment_cols,
            )

            result_df = pd.DataFrame(
                {
                    "Variant": [treatment],
                    "Outcome": [target],
                    "Estimated_Effect_absolute": [estimator.estimate],
                    "CI_lower": [
                        estimator.regression_results.conf_int(alpha=alpha, cols=None)[
                            0
                        ][1]
                    ],
                    "CI_upper": [
                        estimator.regression_results.conf_int(alpha=alpha, cols=None)[
                            1
                        ][1]
                    ],
                    "p_value": [estimator.p_value],
                    "is_significant": [estimator.p_value < alpha],
                }
                # ed.targets,
            )
            result_dfs.append(result_df)

    # concatenate results
    final_df = pd.concat(result_dfs)
    final_df = final_df.set_index(["Variant", "Outcome"])

    return final_df


def run_cuped(ed: ExperimentDataset):
    """
    Applies the CUPED method to estimate treatment effects in experiment.
    Serves as a wrapper to variance reduction methods

    Args:
        ExperimentDataset (ExperimentDataset): An ExperimentDataset object containing the data to be analyzed.

    Returns:
        output (CUPEDOutput): A results object with all fitted estimators that can then be assembled into a results table.
    """
    # check if multiple covariates are to be used
    multivariate = (
        isinstance(ed.pre_experiment_cols, list) and len(ed.pre_experiment_cols) > 1
    )

    # create dummy variables for variants
    dummified_data = pd.get_dummies(ed.data, columns=[ed.variant], drop_first=True)

    # create list of treatments
    treatments = [i for i in range(1, ed.n_variants)]

    estimators = dict()

    # run estimator for each target and treatment
    for target in ed.targets:
        estimators[target] = dict()
        for treatment in treatments:
            # initiate estimator
            if multivariate:
                estimator = MultivariateCUPED()

            else:
                estimator = CUPED()

            estimators[target][treatment] = estimator.fit(
                data=dummified_data,
                treatment_column=ed.variant + "_" + str(treatment),
                target_column=target,
                covariate_column=ed.pre_experiment_cols[0],
                covariate_columns=ed.pre_experiment_cols,
            )

    return CUPEDoutput(estimators=estimators, n_variants=ed.n_variants)


@dataclass
class CUPEDoutput:
    estimators: Dict[str, Dict[str, float]]
    n_variants: int

    def results_table(self, alpha: float = 0.05, has_correction: bool = False):
        # multitest correction NOTE: copied straight from FrequentistTest._alpha_correction
        # TODO: remove duplication, integrate with MultipleHypothesisCorrection
        multitest_correction = (
            "bonferroni" if self.n_variants > 2 and has_correction == "Yes" else None
        )
        if multitest_correction is not None:
            if multitest_correction == "bonferroni":
                correction_level = "per_treatment"  # NOTE: hardcoded for the time being
                n_hypotheses = (self.n_variants - 1) * (
                    correction_level == "per_treatment"
                ) + len(self.targets) * (self.n_variants - 1) * (
                    correction_level == "global"
                )
                alpha_corrected = alpha / n_hypotheses
                alpha = alpha_corrected
            else:
                raise NotImplementedError
        else:
            pass

        result_dfs = []
        for target in self.estimators.keys():
            for treatment in range(1, self.n_variants):
                estimator = self.estimators[target][treatment]
                result_df = pd.DataFrame(
                    {
                        "Variant": [treatment],
                        "Outcome": [target],
                        "Estimated_Effect_absolute": [estimator.estimate],
                        "CI_lower": [
                            estimator.regression_results.conf_int(
                                alpha=alpha, cols=None
                            )[0][1]
                        ],
                        "CI_upper": [
                            estimator.regression_results.conf_int(
                                alpha=alpha, cols=None
                            )[1][1]
                        ],
                        "p_value": [estimator.p_value],
                        "is_significant": [estimator.p_value < alpha],
                    }
                    # ed.targets,
                )
            result_dfs.append(result_df)

        # concatenate results
        final_df = pd.concat(result_dfs)
        final_df = final_df.set_index(["Variant", "Outcome"])
        return final_df


# for debugging frequentist test
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from tw_experimentation.utils import *
    from tw_experimentation.data_generation import *

    rc = RevenueConversion()
    df = rc.generate_data(
        baseline_conversion=0.4,
        treatment_effect_conversion=0.3,
        baseline_mean_revenue=6,
        sigma_revenue=1,
        treatment_effect_revenue=0.5,
    )

    df_abn = rc.generate_data_abn_test(
        n_treatments=2,
        baseline_conversion=0.4,
        treatment_effect_conversion=0.04,
        baseline_mean_revenue=6,
        sigma_revenue=1,
        treatment_effect_revenue=0.5,
    )

    targets = ["conversion", "revenue"]

    metrics = ["binary", "continuous"]

    ed = ExperimentDataset(
        data=df_abn,
        variant="T",
        targets=targets,
        date="trigger_dates",
        metric_types=dict(zip(targets, metrics)),
        n_variants=2,
    )

    ed.preprocess_dataset()
    ft = FrequentistTest(ed=ed)
    ft.compute()
    ft.get_results_table()
