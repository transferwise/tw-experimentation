"""
Tool for testing ab experiments using different approaches for several metrics.
Code contains Tester class with methods for specific tests.
Support only pandas DataFrames now.
"""

import pandas as pd
import numpy as np
from statsmodels import stats as sms
from scipy import stats as scipystats
import statsmodels.stats.weightstats as smws
import matplotlib.pyplot as plt

from sklearn.utils import resample

from typing import List, Union

from SequentialProbabilityRatioTTest import (
    SequentialProbabilityRatioTTest,
    SequentialProbabilityRatioBinaryTest,
)


class BinaryTest:
    def __init__(
        self,
        df: Union[pd.DataFrame, None] = None,
        user_id_column: Union[str, None] = None,
        treatment: Union[str, None] = None,
        metric: Union[str, None] = None,
        method: Union[str, None] = None,
        action_date: Union[str, None] = None,
        cohen_d: Union[float, None] = 0.2,
    ):
        self.df = df
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
        """
        Output contingency table of outcome for treatment and control
        :return: 2x2 DataFrame
        """
        ct = pd.crosstab(self.df[self.treatment], self.df[self.metric])
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
        """
        Fisher's exact test
        :param df: DataFrame
        :param treatment: str
            treatment name
        :param metric: str
            outcome name
        :param direction:
            Options:
                ‘two-sided’: the odds ratio of the underlying population is not one
                ‘less’: the odds ratio of the underlying population is less than one
                ‘greater’: the odds ratio of the underlying population is greater than one
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
            df=self.df,
            treatment=self.treatment,
            outcome=self.metric,
            action_date=self.action_date,
            cohen_d=self.cohen_d,
        ).run_test()


class ContinuousTest:
    def __init__(
        self,
        df: Union[pd.DataFrame, None] = None,
        user_id_column: Union[str, None] = None,
        treatment: Union[str, None] = None,
        metric: Union[str, None] = None,
        method: Union[str, None] = None,
        action_date: Union[str, None] = None,
        cohen_d: Union[float, None] = 0.2,
    ):
        self.df = df
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
            self.df[self.metric].where(self.df[self.treatment] == 1).dropna().mean()
            - self.df[self.metric].where(self.df[self.treatment] == 0).dropna().mean()
        )

    def run(self):
        return eval("self." + self.config_tests[self.method] + "()")

    def mann_whitney_u_test(self):
        """
        Mann-Whitney-U test.
        To be used when sample sized is expected to be skewed / not normally distributed
        :return:
            statistic: float
            pvalue: float
        """
        result = scipystats.mannwhitneyu(
            self.df[self.metric].where(self.df[self.treatment] == 0).dropna(),
            self.df[self.metric].where(self.df[self.treatment] == 1).dropna(),
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
            self.df[self.metric].where(self.df[self.treatment] == 0).dropna(),
            self.df[self.metric].where(self.df[self.treatment] == 1).dropna(),
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
            self.df[self.metric].where(self.df[self.treatment] == 0).dropna(),
            self.df[self.metric].where(self.df[self.treatment] == 1).dropna(),
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
            df=self.df,
            treatment=self.treatment,
            outcome=self.metric,
            action_date=self.action_date,
            cohen_d=self.cohen_d,
        ).run_test()


class ConfidenceIntervals:
    def __init__(
        self,
        df: Union[pd.DataFrame, None] = None,
        user_id_column: Union[str, None] = None,
        treatment: Union[str, None] = None,
        action_date: Union[str, None] = None,
        metric: Union[str, None] = None,
        metric_type: Union[str, None] = "Binary",
        method: Union[str, None] = None,
    ):
        self.df = df
        self.user_id_column = user_id_column
        self.treatment = treatment
        self.action_date = action_date
        self.metric = metric
        self.metric_type = metric_type
        self.method = method
        self.distribution = []

    def bootstrap_confidence_interval(self, alpha=0.05, num_iterations=2000):
        samples = self.bootstrap_samples(n_samples=num_iterations)
        if self.metric_type == "Binary":
            stat_val = BinaryTest(
                df=self.df,
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
                        df=samples[j],
                        user_id_column=self.user_id_column,
                        treatment=self.treatment,
                        metric=self.metric,
                        method=self.method,
                        action_date=self.action_date,
                    ).run()["effect_estimate"]
                )

        elif self.metric_type == "Continuous":
            stat_val = ContinuousTest(
                df=self.df,
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
                        df=samples[j],
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
        samples = []
        for _ in range(n_samples):
            sample = resample(self.df, n_samples=n_elements)
            samples.append(sample)
        return samples

    def bootstrap_treatment_control_samples(self, n_samples=10, n_elements=2000):
        treatment_samples = []
        control_samples = []
        for _ in range(n_samples):
            treatment_sample = resample(
                self.df[self.df[self.treatment] == 1], n_samples=n_elements
            )
            treatment_samples.append(treatment_sample)

            control_sample = resample(
                self.df[self.df[self.treatment] == 0], n_samples=n_elements
            )
            control_samples.append(control_sample)

            samples = treatment_samples + control_samples

        return samples


class Tester:
    """
    Tool for analysing AB tests
    Result includes:
        - Statistical result
    """

    def __init__(
        self,
        df: Union[pd.DataFrame, None] = None,
        user_id_column: Union[str, None] = None,
        treatment: Union[str, None] = None,
        action_date: Union[str, None] = None,
        binary_metrics: Union[List[str], None] = None,
        continuous_metrics: Union[List[str], None] = None,
        customer_features: Union[List[str], None] = None,
    ):
        """
        Tester constructor.
        """
        self.df = df
        self.user_id_column = user_id_column
        self.treatment = treatment
        self.action_date = action_date
        self.binary_metrics = binary_metrics
        self.continuous_metrics = continuous_metrics
        self.customer_features = customer_features

    def data_info(self) -> int:
        return len(self.df)

    def ab_test(
        self,
        method_binary=None,
        method_continuous=None,
        cohen_d=None,
        direction="two-sided",
        alpha: float = 0.05,
        mult_hyp_correction=True,
    ):
        binary_test_results = (
            dict(
                [
                    (
                        metric,
                        BinaryTest(
                            df=self.df,
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
            if not self.binary_metrics is None
            else {}
        )

        continuous_test_results = (
            dict(
                [
                    (
                        metric,
                        ContinuousTest(
                            df=self.df,
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
            if not self.continuous_metrics is None
            else {}
        )

        # multiple testing correction
        #  bonferroni method
        if mult_hyp_correction:
            n_hypotheses = 0
            if not self.binary_metrics is None:
                n_hypotheses += len(self.binary_metrics)
            if not self.continuous_metrics is None:
                n_hypotheses += +len(self.continuous_metrics)

        alpha = alpha / n_hypotheses

        def decision_and_bootstrap(
            results: Union[str, None] = None,
            metrics: Union[list, None] = None,
            metric_type: Union[str, None] = None,
            method: Union[str, None] = None,
        ):
            if not metrics is None:
                for metric in metrics:
                    if (
                        "effect_estimate" in results[metric]
                        and not results[metric]["effect_estimate"] is None
                    ):
                        results[metric]["confidence_interval_bootstrapped"] = (
                            ConfidenceIntervals(
                                df=self.df,
                                user_id_column=self.user_id_column,
                                treatment=self.treatment,
                                metric=metric,
                                metric_type=metric_type,
                                method=method,
                                action_date=self.action_date,
                            ).bootstrap_confidence_interval(alpha=alpha)
                        )
                    if (
                        "pvalue" in results[metric]
                        and not results[metric]["pvalue"] is None
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

        if not self.binary_metrics is None:
            for metric in self.binary_metrics:
                if (
                    "effect_estimate" in binary_test_results[metric]
                    and not binary_test_results[metric]["effect_estimate"] is None
                ):
                    binary_test_results[metric]["confidence_interval_bootstrapped"] = (
                        ConfidenceIntervals(
                            df=self.df,
                            user_id_column=self.user_id_column,
                            treatment=self.treatment,
                            metric=metric,
                            metric_type="Binary",
                            method=method_binary,
                            action_date=self.action_date,
                        ).bootstrap_confidence_interval(alpha=alpha)
                    )
                if (
                    "pvalue" in binary_test_results[metric]
                    and not binary_test_results[metric]["pvalue"] is None
                ):
                    if binary_test_results[metric]["pvalue"] < alpha:
                        binary_test_results[metric]["Decision"] = "Reject H0"
                    else:
                        binary_test_results[metric]["Decision"] = "Accept H0"

        if not self.continuous_metrics is None:
            for metric in self.continuous_metrics:
                if (
                    "effect_estimate" in continuous_test_results[metric]
                    and not continuous_test_results[metric]["effect_estimate"] is None
                ):
                    continuous_test_results[metric][
                        "confidence_interval_bootstrapped"
                    ] = ConfidenceIntervals(
                        df=self.df,
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
                    and not continuous_test_results[metric]["pvalue"] is None
                ):
                    if continuous_test_results[metric]["pvalue"] < alpha:
                        continuous_test_results[metric]["Decision"] = "Reject H0"
                    else:
                        continuous_test_results[metric]["Decision"] = "Accept H0"

        return {**binary_test_results, **continuous_test_results}


def multiple_hypothesis_correction(pvals, method="bonferroni"):
    correction = sms.multitest.multipletests(pvals=pvals, method=method)
    return correction
