"""
Tool to set up experiments: alpha, beta, MDE, power
Code contains Setuper class with methods.
"""

import pandas as pd
import numpy as np

from typing import List, Union
from dataclasses import dataclass

import statsmodels.api as sm
import statsmodels.formula.api as sm
import statsmodels.stats as stats
from statsmodels.stats import power as pwr

import plotly.express as px

from tw_experimentation.utils import ExperimentDataset


@dataclass
class ExpDesignAutoCalculate:
    """
    A class for automatically calculating standard deviation and mean values
    for pre-experiment columns in an ExperimentDataset.

    Attributes:
        ed (ExperimentDataset): The ExperimentDataset object containing
            the pre-experiment data.
    """

    ed: ExperimentDataset

    def __post_init__(self):
        self.sds = {
            target_pre_exp: self.ed.data[target_pre_exp].std()
            for target_pre_exp in self.pre_experiment_cols
        }

        self.means = {
            target_pre_exp: self.ed.data[target_pre_exp].mean()
            for target_pre_exp in self.pre_experiment_cols
        }

    @property
    def metric_types(self):
        return self.ed.metric_types

    @property
    def pre_experiment_cols(self):
        """
        Returns:
            List[str]: A list of pre-experiment column names.
        """
        return self.ed.pre_experiment_cols

    def sd(self, target_pre_exp):
        """
        Args:
            target_pre_exp (str): The name of the pre-experiment column.

        Returns:
            float: The standard deviation value for the given pre-experiment column.
        """
        return self.sds[target_pre_exp]

    def mean(self, target_pre_exp):
        """
        Args:
            target_pre_exp (str): The name of the pre-experiment column.

        Returns:
            float: The mean value for the given pre-experiment column.
        """
        return self.means[target_pre_exp]


class Setuper:
    """Tool for designing AB tests
    Result includes:
        - Minimal Detectable Effect Size
        - Sample Size calculation
        - Sample Size vs MDE plot
    """

    def __init__(
        self,
        alpha=0.05,
        beta=0.2,
        treatment_share=0.5,
        effect_size=0.02,
        alternative="two-sided",
    ) -> None:
        """

        Args:
            alpha (float, optional): Type-I error. Defaults to .05.
            beta (float, optional): Type-II error. Defaults to .2.
            treatment_share (float, optional): Share of treatment group in the sample. Defaults to .5.
            effect_size (float, optional): Standardized effect size: Cohen d. Defaults to .02.
            alternative (str, optional): 'two_sided', 'larger', 'smaller'
            for alternative test designs. Defaults to 'two-sided'.
        """

        self.alpha = alpha
        self.beta = beta
        self.treatment_share = treatment_share
        self.effect_size = effect_size
        self.ratio = treatment_share / (1 - treatment_share)
        self.alternative = alternative

    @classmethod
    def from_uplift(
        cls,
        uplift,
        sd,
        mean,
        alpha=0.05,
        beta=0.2,
        treatment_share=0.5,
        relation="absolute",
    ):
        """Design test size from uplift instead of standardized effect size

        Args:
            uplift (float): uplift to detect
            sd (float): standard deviation
            mean (float): expected baseline mean
            alpha (float, optional): type-I error. Defaults to .05.
            beta (float, optional): type-II error. Defaults to .2.
            treatment_share (float, optional): treatment share in population. Defaults to .5.
            relation (str, optional): 'two-sided', 'larger' or 'smaller. Defaults to 'absolute'.

        Returns:
            Setuper class instance: Setuper class with uplift converted to effect size
        """
        return cls(
            alpha,
            beta,
            treatment_share,
            uplift_to_effect_size(uplift, mean, sd, relation),
        )

    def sample_size_t_test(self):
        """t test sample size calculation for continuous outcomes

        Returns:
            dict: sample sizes per group
        """
        hyp_test = getattr(stats.power, "tt_ind_solve_power")
        treat_control_ratio = (1 - self.treatment_share) / self.treatment_share
        treatment_ss = hyp_test(
            effect_size=self.effect_size,
            alpha=self.alpha,
            power=1 - self.beta,
            ratio=treat_control_ratio,
        )
        total_ss = int(treatment_ss) + int(treatment_ss + treat_control_ratio)
        return {
            "Total Sample Size": int(total_ss),
            "Treatment Sample Size": int(treatment_ss),
            "Control Sample Size": int(treatment_ss * treat_control_ratio),
        }

    def sample_size_two_sample_proportion_z_test(self):
        """two sample proportion z test sample size calculation

        Returns:
            dict: sample sizes per group
        """
        hyp_test = getattr(stats.power, "zt_ind_solve_power")
        treat_control_ratio = (1 - self.treatment_share) / self.treatment_share
        treatment_ss = hyp_test(
            effect_size=self.effect_size,
            alpha=self.alpha,
            power=1 - self.beta,
            ratio=treat_control_ratio,
        )
        total_ss = treatment_ss * (1 + treat_control_ratio)
        return {
            "Total Sample Size": int(total_ss),
            "Treatment Sample Size": int(treatment_ss),
            "Control Sample Size": int(treatment_ss * treat_control_ratio),
        }

    def sample_size_chi_squared_test(self):
        """
        Sample Size calculation for chi squared test
        statsmodels.stats.proportion.proportions_chisquare
        """
        pass

    def sample_size_proportion_test(
        self, uplift, baseline_conversion=0.5, relation="absolute"
    ):
        """Sample sie for proportion test, requires uplift and
        automatically calculates standard deviation

        Args:
            uplift (float): uplift to detect
            baseline_conversion (float, optional): baseline conversion. Defaults to .5.
            relation (str, optional): 'two-sided', 'larger' or 'smaller'. Defaults to 'absolute'.

        Returns:
            float: total sample size
        """

        if relation == "absolute":
            pass
        elif relation == "relative":
            uplift = uplift * baseline_conversion

        return (
            stats.proportion.samplesize_proportions_2indep_onetail(
                diff=uplift,
                prop2=baseline_conversion,
                power=1 - self.beta,
                alpha=self.alpha,
                alternative=self.alternative,
            )
            / self.treatment_share
        )

    def effect_size_proportion_test(self):
        """effect size for statsmodels.stats.proportion.test_proportions_2indep"""
        pass

    def effect_size_t_test(self, nobs):
        """effect size for t test given power and number of observations

        Args:
            nobs (int): number of observations

        Returns:
            float: minimal detectable effect size
        """
        nobs1 = nobs * self.treatment_share
        ratio = self.ratio
        analysis = pwr.TTestIndPower()
        esresult = analysis.solve_power(
            power=1 - self.beta, nobs1=nobs1, ratio=ratio, alpha=self.alpha
        )
        return esresult

    def effect_size_two_sample_z_test(self, nobs):
        """effect size of two sample independent proportion z test

        Args:
            nobs (int): number of observations

        Returns:
            float: minimal detectable effect size
        """

        hyp_test = getattr(stats.power, "zt_ind_solve_power")
        nobs1 = nobs * self.treatment_share
        treat_control_ratio = (1 - self.treatment_share) / self.treatment_share
        mde = hyp_test(
            nobs1=nobs1,
            alpha=self.alpha,
            power=1 - self.beta,
            ratio=treat_control_ratio,
        )
        return mde

    def effect_size_chi_squared_test(self):
        """TODO: MDE for chi-squared test"""
        pass

    def power_t_test(self, nobs):
        """power of t test for continuous outcomes

        Args:
            nobs (int): number of observations

        Returns:
            float: power
        """
        ratio = self.ratio
        analysis = pwr.TTestIndPower()
        pwresult = analysis.solve_power(
            effect_size=self.effect_size,
            power=None,
            alpha=self.alpha,
            nobs1=nobs,
            ratio=ratio,
        )
        return pwresult

    def power_two_sample_z_test(self, nobs):
        """power two sample independent proportion z test

        Args:
            nobs (int): number of observations

        Returns:
            float: power
        """
        hyp_test = getattr(stats.power, "zt_ind_solve_power")
        nobs1 = nobs * self.treatment_share
        treat_control_ratio = (1 - self.treatment_share) / self.treatment_share
        power = hyp_test(
            nobs1=nobs1,
            alpha=self.alpha,
            effect_size=self.effect_size,
            ratio=treat_control_ratio,
        )
        return power

    def power_proportion_test(
        self, uplift, nobs, baseline_conversion=0.5, relation="absolute"
    ):
        """power proportion test (as in evan miller)

        Args:
            uplift (float): uplift to detect
            nobs (int): number of observations
            baseline_conversion (float, optional): baseline conversion rate. Defaults to .5.
            relation (str, optional): 'two-sided', 'larger' or 'smaller'. Defaults to 'absolute'.

        Returns:
            float: power
        """

        if relation == "absolute":
            pass
        elif relation == "relative":
            uplift = uplift * baseline_conversion

        nobs1 = nobs * self.treatment_share

        return stats.proportion.power_proportions_2indep(
            diff=uplift,
            nobs1=nobs1,
            ratio=self.ratio,
            prop2=baseline_conversion,
            alpha=self.alpha,
            alternative=self.alternative,
        )

    def plot_sample_size_to_mde(
        self,
        effect_size_function,
        max_sample_size=1000,
    ):
        """plot sample size to MDE

        Args:
            effect_size_function (function): function that maps sample size to MDE
            max_sample_size (int, optional): x axis limit. Defaults to 1000.
        """
        # TODO: Also plot to sample size to uplift
        sample_size = np.arange(200, max_sample_size, 10)
        mde = np.array(list(map(effect_size_function, sample_size)))
        fig = px.line(
            x=sample_size,
            y=mde,
            width=600,
            height=600,
            title="MDE per Sample Size",
            labels={"x": "Sample Size", "y": "MDE"},
        )
        fig.show()


def uplift_to_effect_size(uplift, mean, sd, relation="absolute"):
    """Convert uplift to effect size

    Args:
        uplift (float): expected uplift
        mean (float): estimated mean
        sd (float): estimated standard deviation

    Returns:
        float: effect size for power calculation
    """

    if relation == "absolute":
        es = uplift / sd
    elif relation == "relative":
        es = (uplift * mean) / sd
    else:
        raise Exception('Uplfit relation must be "relative" or "abosolut"')
    return es


def effect_size_to_uplift(es, mean, sd, relation="absolute"):
    """Convert effect size to uplift

    Args:
        es (float): expected effect size
        mean (float): estimated mean
        sd (float): estimated standard deviation

    Returns:
        float: uplift for power calculation
    """
    if relation == "absolute":
        uplift = es * sd
    elif relation == "relative":
        uplift = es * (sd / mean)
    else:
        raise Exception('Uplfit relation must be "relative" or "abosolut"')
    return uplift
