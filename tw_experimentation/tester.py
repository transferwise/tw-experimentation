"""Tool for testing ab experiments using different approaches for several metrics.

Code contains Tester class with methods for specific tests. Support only pandas
DataFrames now.
"""

from typing import List, Union

import numpy as np
import pandas as pd
from scipy import stats as scipystats
from SequentialProbabilityRatioTTest import (
    SequentialProbabilityRatioBinaryTest,
    SequentialProbabilityRatioTTest,
)
from sklearn.utils import resample
from statsmodels import stats as sms


class BinaryTest:
    def __init__(
        self,
        experiment_df: Union[pd.DataFrame, None] = None,
        user_id_column: Union[str, None] = None,
        treatment: Union[str, None] = None,
        metric: Union[str, None] = None,
        method: Union[str, None] = None,
        action_date: Union[str, None] = None,
        cohen_d: Union[float, None] = 0.2,
    ):
        self.experiment_df = experiment_df
        self.user_id_column = user_id_column
        self.treatment = treatment
        self.metric = metric
        self.method = method
        self.action_date = action_date
        self.cohen_d = cohen_d
        self.ct = self.contingency_table()
        self.config_tests = {
            "Proportion": "static_proportion_test",
            "Fisher": "fishers_exact_test",
            "Chi": "chi_squared_test",
            "Dynamic": "dynamic_test",
        }

    def contingency_table(self):
        """Output contingency table of outcome for treatment and control :return: 2x2
        DataFrame."""
        ct = pd.crosstab(
            self.experiment_df[self.treatment], self.experiment_df[self.metric]
        )
        idx = pd.Index(["Control", "Treatment"], name=None)
        ct.index = idx
        return ct

    def effect_estimate(self):
        return (self.ct.iloc[1, 1] / self.ct.iloc[1, :].sum()) - (
            self.ct.iloc[0, 1] / self.ct.iloc[0, :].sum()
        )

    def run(self):
        return eval("self." + self.config_tests[self.method] + "()")

    def static_proportion_test(self, direction="two-sided"):
        ct = self.ct
        result = sms.proportion.test_proportions_2indep(
            count1=ct.iloc[0, 1],
            nobs1=ct.iloc[0].sum(),
            count2=ct.iloc[1, 1],
            nobs2=ct.iloc[1].sum(),
            alternative=direction,
        )
        return {
            "statistic": result.statistic,
            "pvalue": result.pvalue,
            "variance": result.variance,
            "effect_estimate": self.effect_estimate(),
            "Decision": None,
            "further_results": {
                "contingency table": self.ct,
                "odds_ratio": result.odds_ratio,
                "ratio": result.ratio,
            },
            "plots": None,
        }

    def fishers_exact_test(self, direction="two-sided"):
        """Fisher's exact test :param experiment_df: DataFrame :param treatment:
            str treatment name
        :param metric: str outcome name :param direction:

        Options:
                ‘two-sided’: the odds ratio of the underlying population is not one
                ‘less’: the odds ratio of the underlying population is less than one
                ‘greater’: the odds ratio of the underlying population is greater than 1
        :return: SignificanceResult
                    statistic: float
                    pvalue: float
        """
        ct = self.ct
        result = scipystats.fisher_exact(ct, alternative=direction)
        return {
            "statistic": result[0],
            "pvalue": result[1],
            "variance": None,
            "effect_estimate": self.effect_estimate(),
            "Decision": None,
            "further_results": None,
            "plots": None,
        }

    def chi_squared_test(self):
        """
        :return:
            chi2stat: float
            p-value: float
        """
        ct = self.ct
        result = scipystats.chi2_contingency(ct)[:3]
        return {
            "statistic": result[0],
            "pvalue": result[1],
            "further_results": {"degrees_of_freedom": result[2]},
        }

    def dynamic_test(self):
        return SequentialProbabilityRatioBinaryTest(
            experiment_df=self.experiment_df,
            treatment=self.treatment,
            outcome=self.metric,
            action_date=self.action_date,
            cohen_d=self.cohen_d,
        ).run_test()


class ContinuousTest:
    def __init__(
        self,
        experiment_df: Union[pd.DataFrame, None] = None,
        user_id_column: Union[str, None] = None,
        treatment: Union[str, None] = None,
        metric: Union[str, None] = None,
        method: Union[str, None] = None,
        action_date: Union[str, None] = None,
        cohen_d: Union[float, None] = 0.2,
    ):
        self.experiment_df = experiment_df
        self.user_id_column = user_id_column
        self.treatment = treatment
        self.metric = metric
        self.method = method
        self.action_date = action_date
        self.cohen_d = cohen_d
        self.config_tests = {
            "Mann-Whitney": "mann_whitney_u_test",
            "Ttest": "ttest",
            "Welch_Ttest": "welch_ttest",
            "Dynamic": "dynamic_test",
        }

    def effect_estimate(self):
        return (
            self.experiment_df[self.metric]
            .where(self.experiment_df[self.treatment] == 1)
            .dropna()
            .mean()
            - self.experiment_df[self.metric]
            .where(self.experiment_df[self.treatment] == 0)
            .dropna()
            .mean()
        )

    def run(self):
        return eval("self." + self.config_tests[self.method] + "()")

    def mann_whitney_u_test(self):
        """Mann-Whitney-U test.

        To be used when sample sized is expected to be skewed /
            not normally distributed
        :return:
            statistic: float
            pvalue: float
        """
        result = scipystats.mannwhitneyu(
            self.experiment_df[self.metric]
            .where(self.experiment_df[self.treatment] == 0)
            .dropna(),
            self.experiment_df[self.metric]
            .where(self.experiment_df[self.treatment] == 1)
            .dropna(),
        )

        return {
            "statistic": result[0],
            "pvalue": result[1],
            "variance": None,
            "effect_estimate": self.effect_estimate(),
            "Decision": None,
            "further_results": None,
            "plots": None,
        }

    def ttest(self):
        result = sms.weightstats.ttest_ind(
            self.experiment_df[self.metric]
            .where(self.experiment_df[self.treatment] == 0)
            .dropna(),
            self.experiment_df[self.metric]
            .where(self.experiment_df[self.treatment] == 1)
            .dropna(),
        )
        return {
            "statistic": result[0],
            "pvalue": result[1],
            "variance": None,
            "effect_estimate": self.effect_estimate(),
            "Decision": None,
            "further_results": {"degrees_of_freedom": result[2]},
            "plots": None,
        }

    def welch_ttest(self):
        result = sms.weightstats.ttest_ind(
            self.experiment_df[self.metric]
            .where(self.experiment_df[self.treatment] == 0)
            .dropna(),
            self.experiment_df[self.metric]
            .where(self.experiment_df[self.treatment] == 1)
            .dropna(),
            usevar="unequal",
        )
        return {
            "statistic": result[0],
            "pvalue": result[1],
            "variance": None,
            "effect_estimate": self.effect_estimate(),
            "Decision": None,
            "further_results": {"degrees_of_freedom": result[2]},
            "plots": None,
        }

    def dynamic_test(self):
        return SequentialProbabilityRatioTTest(
            experiment_df=self.experiment_df,
            treatment=self.treatment,
            outcome=self.metric,
            action_date=self.action_date,
            cohen_d=self.cohen_d,
        ).run_test()


class ConfidenceIntervals:
    """
    Class for calculating confidence intervals for experiment metrics.

    Args:
        experiment_df (pd.DataFrame or None): The experiment
            data as a pandas DataFrame.
        user_id_column (str or None): The column name for the user ID.
        treatment (str or None): The column name for the treatment variable.
        action_date (str or None): The column name for the action date.
        metric (str or None): The column name for the metric variable.
        metric_type (str or None): The type of metric. Defaults to "Binary".
        method (str or None): The method used for analysis.

    Attributes:
        experiment_df (pd.DataFrame or None): The experiment
            data as a pandas DataFrame.
        user_id_column (str or None): The column name for the user ID.
        treatment (str or None): The column name for the treatment variable.
        action_date (str or None): The column name for the action date.
        metric (str or None): The column name for the metric variable.
        metric_type (str or None): The type of metric. Defaults to "Binary".
        method (str or None): The method used for analysis.
        distribution (list): List to store the distribution of effect estimates.

    Methods:
        bootstrap_confidence_interval(alpha=0.05, num_iterations=2000):
            Calculates the bootstrap confidence interval for
                the effect estimate.
        bootstrap_samples(n_samples=10, n_elements=2000):
            Generates bootstrap samples from the experiment data.
        bootstrap_treatment_control_samples(n_samples=10, n_elements=2000):
            Generates bootstrap samples for treatment and control groups.

    """

    def __init__(
        self,
        experiment_df: Union[pd.DataFrame, None] = None,
        user_id_column: Union[str, None] = None,
        treatment: Union[str, None] = None,
        action_date: Union[str, None] = None,
        metric: Union[str, None] = None,
        metric_type: Union[str, None] = "Binary",
        method: Union[str, None] = None,
    ):
        """
        Initializes the ConfidenceIntervals class.

        Args:
            experiment_df (pd.DataFrame or None): The experiment
                data as a pandas DataFrame.
            user_id_column (str or None): The column name for the user ID.
            treatment (str or None): The column name for
                the treatment variable.
            action_date (str or None): The column name for the action date.
            metric (str or None): The column name for the metric variable.
            metric_type (str or None): The type of metric. Defaults to "Binary".
            method (str or None): The method used for analysis.

        Returns:
            None
        """
        self.experiment_df = experiment_df
        self.user_id_column = user_id_column
        self.treatment = treatment
        self.action_date = action_date
        self.metric = metric
        self.metric_type = metric_type
        self.method = method
        self.distribution = []

    def bootstrap_confidence_interval(self, alpha=0.05, num_iterations=2000):
        """
        Calculates the bootstrap confidence interval for the effect estimate.

        Args:
            alpha (float): The significance level. Defaults to 0.05.
            num_iterations (int): The number of bootstrap iterations.
                Defaults to 2000.

        Returns:
            tuple: The lower and upper bounds of the confidence interval.
        """
        samples = self.bootstrap_samples(n_samples=num_iterations)
        if self.metric_type == "Binary":
            stat_val = BinaryTest(
                experiment_df=self.experiment_df,
                user_id_column=self.user_id_column,
                treatment=self.treatment,
                metric=self.metric,
                method=self.method,
                action_date=self.action_date,
            ).run()["effect_estimate"]
            distribution = []
            for j in range(num_iterations):
                distribution.append(
                    BinaryTest(
                        experiment_df=samples[j],
                        user_id_column=self.user_id_column,
                        treatment=self.treatment,
                        metric=self.metric,
                        method=self.method,
                        action_date=self.action_date,
                    ).run()["effect_estimate"]
                )

        elif self.metric_type == "Continuous":
            stat_val = ContinuousTest(
                experiment_df=self.experiment_df,
                user_id_column=self.user_id_column,
                treatment=self.treatment,
                metric=self.metric,
                method=self.method,
                action_date=self.action_date,
            ).run()["effect_estimate"]
            distribution = []
            for j in range(num_iterations):
                distribution.append(
                    ContinuousTest(
                        experiment_df=samples[j],
                        user_id_column=self.user_id_column,
                        treatment=self.treatment,
                        metric=self.metric,
                        method=self.method,
                        action_date=self.action_date,
                    ).run()["effect_estimate"]
                )

        self.distribution = distribution

        c_low = 2 * stat_val - np.percentile(distribution, 100 * (1 - alpha / 2.0))
        c_high = 2 * stat_val - np.percentile(distribution, 100 * (alpha / 2.0))

        return (c_low, c_high)

    def bootstrap_samples(self, n_samples=10, n_elements=2000):
        """
        Generates bootstrap samples from the experiment data.

        Args:
            n_samples (int): The number of bootstrap samples to generate.
                Defaults to 10.
            n_elements (int): The number of elements in each bootstrap sample.
                Defaults to 2000.

        Returns:
            list: List of bootstrap samples.
        """
        samples = []
        for _ in range(n_samples):
            sample = resample(self.experiment_df, n_samples=n_elements)
            samples.append(sample)
        return samples

    def bootstrap_treatment_control_samples(self, n_samples=10, n_elements=2000):
        """
        Generates bootstrap samples for treatment and control groups.

        Args:
            n_samples (int): The number of bootstrap samples to generate.
                Defaults to 10.
            n_elements (int): The number of elements in each bootstrap sample.
                Defaults to 2000.

        Returns:
            list: List of bootstrap samples for treatment and control groups.
        """
        treatment_samples = []
        control_samples = []
        for _ in range(n_samples):
            treatment_sample = resample(
                self.experiment_df[self.experiment_df[self.treatment] == 1],
                n_samples=n_elements,
            )
            treatment_samples.append(treatment_sample)

            control_sample = resample(
                self.experiment_df[self.experiment_df[self.treatment] == 0],
                n_samples=n_elements,
            )
            control_samples.append(control_sample)

            samples = treatment_samples + control_samples

        return samples


class Tester:
    """Tool for analysing AB tests.

    Result includes:
        - Statistical result
    """

    def __init__(
        self,
        experiment_df: Union[pd.DataFrame, None] = None,
        user_id_column: Union[str, None] = None,
        treatment: Union[str, None] = None,
        action_date: Union[str, None] = None,
        binary_metrics: Union[List[str], None] = None,
        continuous_metrics: Union[List[str], None] = None,
        customer_features: Union[List[str], None] = None,
    ):
        """Tester constructor."""
        self.experiment_df = experiment_df
        self.user_id_column = user_id_column
        self.treatment = treatment
        self.action_date = action_date
        self.binary_metrics = binary_metrics
        self.continuous_metrics = continuous_metrics
        self.customer_features = customer_features

    def data_info(self) -> int:
        return len(self.experiment_df)

    def ab_test(
        self,
        method_binary=None,
        method_continuous=None,
        cohen_d=None,
        alpha: float = 0.05,
        mult_hyp_correction=True,
    ):
        binary_test_results = (
            dict(
                [
                    (
                        metric,
                        BinaryTest(
                            experiment_df=self.experiment_df,
                            user_id_column=self.user_id_column,
                            treatment=self.treatment,
                            metric=metric,
                            method=method_binary,
                            action_date=self.action_date,
                            cohen_d=cohen_d,
                        ).run(),
                    )
                    for metric in self.binary_metrics
                ]
            )
            if self.binary_metrics is not None
            else {}
        )

        continuous_test_results = (
            dict(
                [
                    (
                        metric,
                        ContinuousTest(
                            experiment_df=self.experiment_df,
                            user_id_column=self.user_id_column,
                            treatment=self.treatment,
                            metric=metric,
                            method=method_continuous,
                            action_date=self.action_date,
                            cohen_d=cohen_d,
                        ).run(),
                    )
                    for metric in self.continuous_metrics
                ]
            )
            if self.continuous_metrics is not None
            else {}
        )

        # multiple testing correction
        #  bonferroni method
        if mult_hyp_correction:
            n_hypotheses = 0
            if self.binary_metrics is not None:
                n_hypotheses += len(self.binary_metrics)
            if self.continuous_metrics is not None:
                n_hypotheses += +len(self.continuous_metrics)

        alpha = alpha / n_hypotheses

        def decision_and_bootstrap(
            results: Union[str, None] = None,
            metrics: Union[list, None] = None,
            metric_type: Union[str, None] = None,
            method: Union[str, None] = None,
        ):
            if metrics is not None:
                for metric in metrics:
                    if (
                        "effect_estimate" in results[metric]
                        and results[metric]["effect_estimate"] is not None
                    ):
                        results[metric][
                            "confidence_interval_bootstrapped"
                        ] = ConfidenceIntervals(
                            experiment_df=self.experiment_df,
                            user_id_column=self.user_id_column,
                            treatment=self.treatment,
                            metric=metric,
                            metric_type=metric_type,
                            method=method,
                            action_date=self.action_date,
                        ).bootstrap_confidence_interval(
                            alpha=alpha
                        )
                    if (
                        "pvalue" in results[metric]
                        and results[metric]["pvalue"] is not None
                    ):
                        if results[metric]["pvalue"] < alpha:
                            results[metric]["Decision"] = "Reject H0"
                        else:
                            results[metric]["Decision"] = "Accept H0"

        decision_and_bootstrap(
            binary_test_results, self.binary_metrics, "Binary", method_binary
        )
        decision_and_bootstrap(
            continuous_test_results,
            self.continuous_metrics,
            "Continuous",
            method_continuous,
        )

        if self.binary_metrics is not None:
            for metric in self.binary_metrics:
                if (
                    "effect_estimate" in binary_test_results[metric]
                    and binary_test_results[metric]["effect_estimate"] is not None
                ):
                    binary_test_results[metric][
                        "confidence_interval_bootstrapped"
                    ] = ConfidenceIntervals(
                        experiment_df=self.experiment_df,
                        user_id_column=self.user_id_column,
                        treatment=self.treatment,
                        metric=metric,
                        metric_type="Binary",
                        method=method_binary,
                        action_date=self.action_date,
                    ).bootstrap_confidence_interval(
                        alpha=alpha
                    )
                if (
                    "pvalue" in binary_test_results[metric]
                    and binary_test_results[metric]["pvalue"] is not None
                ):
                    if binary_test_results[metric]["pvalue"] < alpha:
                        binary_test_results[metric]["Decision"] = "Reject H0"
                    else:
                        binary_test_results[metric]["Decision"] = "Accept H0"

        if self.continuous_metrics is not None:
            for metric in self.continuous_metrics:
                if (
                    "effect_estimate" in continuous_test_results[metric]
                    and continuous_test_results[metric]["effect_estimate"] is not None
                ):
                    continuous_test_results[metric][
                        "confidence_interval_bootstrapped"
                    ] = ConfidenceIntervals(
                        experiment_df=self.experiment_df,
                        user_id_column=self.user_id_column,
                        treatment=self.treatment,
                        metric=metric,
                        metric_type="Continuous",
                        method=method_continuous,
                        action_date=self.action_date,
                    ).bootstrap_confidence_interval(
                        alpha=alpha
                    )
                if (
                    "pvalue" in continuous_test_results[metric]
                    and continuous_test_results[metric]["pvalue"] is not None
                ):
                    if continuous_test_results[metric]["pvalue"] < alpha:
                        continuous_test_results[metric]["Decision"] = "Reject H0"
                    else:
                        continuous_test_results[metric]["Decision"] = "Accept H0"

        return {**binary_test_results, **continuous_test_results}


def multiple_hypothesis_correction(pvals, method="bonferroni"):
    """Multiple hypothesis correction.

    Args:
        pvals (list): p values
        method (str, optional): multiple hypothesis correction method.
            Defaults to "bonferroni".

    Returns:
        _type_: corrected values
    """
    correction = sms.multitest.multipletests(pvals=pvals, method=method)
    return correction
