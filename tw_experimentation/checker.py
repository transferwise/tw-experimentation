import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import plotly.graph_objects as go

from scipy.stats import shapiro
from scipy.stats.contingency import expected_freq
import statsmodels.api as sm
from itertools import product

from dataclasses import dataclass, field

from typing import List, Dict, Union
import copy
from tw_experimentation.utils import (
    variant_name_map,
    ExperimentDataset,
)
from tw_experimentation.plotting.monitoring_plots import (
    fig_variant_segment_dependence,
    plot_sample_size_pie,
    plot_dynamic_sample_size,
    target_metric_distribution,
    plot_target_metric_cdf,
    plot_segment_sample_size,
    plot_segment_histograms,
    plot_sequential_test,
    plot_qq_normal,
    plot_qq_variants,
)


@dataclass
class Monitoring:
    ed: ExperimentDataset

    def __post_init__(self):
        assert self.ed.is_preprocessed
        self.sample_size_now = self.ed.sample_sizes
        self.total_sample_size = self.ed.total_sample_size

        self.nonbinary_targets = [
            target
            for target in self.ed.targets
            if self.ed.metric_types[target] in ["discrete", "continuous"]
        ]

    def _dynamic_sample_size_descriptives(self):
        assert self.ed.is_dynamic_observation
        df_base = copy.deepcopy(self.ed.data).sort_values(
            self.ed.date, ignore_index=True
        )
        cols_relevant = [self.ed.variant, self.ed.date]
        df_dyn_avg = df_base[cols_relevant]

        df_dyn_avg["variant_cnt"] = df_base.groupby(
            self.ed.variant,
            group_keys=False,
        )[
            self.ed.date
        ].apply(lambda x: x.expanding().count())

        return df_dyn_avg

    @property
    def sample_size_table(self):
        """
        Returns a pandas DataFrame containing the sample sizes for each variant and the total sample size.

        Returns:
            pandas.DataFrame: A DataFrame with the sample sizes for each variant and the total sample size.
                The index contains the variant names and "Total sample size", and the only column is "Sample size".
        """
        SAMPLE_SIZE_TITLE = "Sample size"
        sample_size_now = self.ed.sample_sizes
        variant_names = variant_name_map(self.ed.n_variants)
        assert list(variant_names.keys()) == list(sample_size_now.keys())
        df_sample_size_now = pd.DataFrame(
            sample_size_now.values(),
            columns=[SAMPLE_SIZE_TITLE],
            index=list(variant_names.values()),
        )
        df_sample_size_now.columns = [SAMPLE_SIZE_TITLE]
        sample_size_total = pd.DataFrame(
            self.total_sample_size,
            columns=[SAMPLE_SIZE_TITLE],
            index=["Total sample size"],
        )

        sample_size_table = pd.concat([df_sample_size_now, sample_size_total])
        return sample_size_table

    def _qq_variant_variant(self, target):
        quantiles = np.linspace(1, 100, num=200)

        qq_variants = {
            variant: np.percentile(
                self.ed.data[self.ed.data[self.ed.variant] == variant][target],
                quantiles,
            )
            for variant in range(self.ed.n_variants)
        }
        return qq_variants

    def qqplot_variant_variant(self, target):
        qq_variants = self._qq_variant_variant(target)
        return plot_qq_variants(qq_variants, self.ed, target)

    def create_tables_and_plots(self):
        """
        Creates the tables and plots for the monitoring.
        """
        fig_sample_size_pie = plot_sample_size_pie(self.ed)

        if self.ed.is_dynamic_observation:
            df_dynamic_descriptive = self._dynamic_sample_size_descriptives()
            fig_dynamic_sample_size = plot_dynamic_sample_size(
                df_dynamic_descriptive, self.ed
            )
        else:
            fig_dynamic_sample_size = None
        fig_target_metric_distribution = {
            target: target_metric_distribution(self.ed, target)
            for target in self.ed.targets
        }

        fig_target_cdf = {
            target: plot_target_metric_cdf(self.ed, target)
            for target in self.nonbinary_targets
        }

        fig_target_qq_variants = {
            target: self.qqplot_variant_variant(target)
            for target in self.nonbinary_targets
        }

        return MonitoringOutput(
            sample_size_table=self.sample_size_table,
            fig_sample_size_pie=fig_sample_size_pie,
            fig_dynamic_sample_size=fig_dynamic_sample_size,
            fig_target_metric_distribution=fig_target_metric_distribution,
            fig_target_cdf=fig_target_cdf,
            fig_target_qq_variants=fig_target_qq_variants,
        )


@dataclass
class NormalityChecks:
    ed: ExperimentDataset
    """
    A class for checking whether metrics regression residuals are normally distributed.
    Relevant for decision to run t-test or not.

    Attributes:
    -----------
    ed : ExperimentData
        An instance of the ExperimentData class containing the experiment data.
    relevant_targets : list
        A list of relevant targets for the experiment. Binary targets are excluded.
    standardized_residuals : dict
        A dictionary containing standardized residuals for each relevant target and variant.

    Methods:
    --------
    __post_init__(self)
        Calculates standardized residuals for each relevant target and variant.
    """

    def __post_init__(self):
        """
        Calculates standardized residuals for each relevant target and variant.
        """
        self.relevant_targets = [
            target
            for target in self.ed.targets
            if self.ed.metric_types[target] in ["discrete", "continuous"]
        ]

        self.standardized_residuals = {target: {} for target in self.relevant_targets}

        for target, variant in product(
            self.relevant_targets, range(1, self.ed.n_variants)
        ):
            data = self.ed.data[self.ed.data[self.ed.variant] == variant]
            exog = sm.add_constant(data[self.ed.variant])
            mod_fit = sm.OLS(data[target], exog).fit()
            influence = mod_fit.get_influence()
            standardized_residuals = influence.resid_studentized_internal
            self.standardized_residuals[target][variant] = standardized_residuals

    def qqplot(self, target):
        """Get a quantile-quantile plot for a given target metric"""
        assert target in self.ed.targets and self.ed.metric_types[target] in [
            "continuous",
            "discrete",
        ]
        fig = plot_qq_normal(self.ed, target, self.standardized_residuals)
        return fig

    def all_qqplots(self):
        """Get Q-Q plots for all relevant target metrics"""
        return {target: self.qqplot(target) for target in self.relevant_targets}

    def shapiro_wilk_test(self, target, alpha=0.05):
        """Perform Shapiro-Wilk test for normality on a given target metric"""
        results = {"variant": [], "statistic": [], "p-value": []}
        variant_names = variant_name_map(self.ed.n_variants)
        for variant in range(1, self.ed.n_variants):
            stat, p_value = shapiro(self.standardized_residuals[target][variant])
            results["variant"].append(variant_names[variant])
            results["statistic"].append(stat)
            results["p-value"].append(p_value)
            interpretation = (
                "t-test can be used (sample looks normal)"
                if p_value > alpha
                else (
                    "t-test assumptions likely to be violated, consider other methods"
                    " (e.g. Bayesian)"
                )
            )
            results["Decision"] = interpretation
            results_df = pd.DataFrame.from_dict(results).set_index("variant")
        return results_df

    def all_shapiro_wilk_tests(self, alpha=0.05):
        """Perform Shapiro-Wilk test for normality on all relevant target metrics"""
        return {
            target: self.shapiro_wilk_test(target, alpha)
            for target in self.relevant_targets
        }

    def create_results(self, alpha=0.05):
        """
        Creates an object containing the relevant targets, QQ plots, and Shapiro-Wilk test results.

        Args:
            alpha (float): The significance level for the Shapiro-Wilk test. Defaults to 0.05.

        Returns:
            NormalityChecksOutput: An object containing the relevant targets, QQ plots, and Shapiro-Wilk test results.
        """
        figs_qqplots = self.all_qqplots()
        tables_shapiro_wilk = self.all_shapiro_wilk_tests(alpha=alpha)
        return NormalityChecksOutput(
            targets=self.relevant_targets,
            figs_qqplots=figs_qqplots,
            tables_shapiro_wilk=tables_shapiro_wilk,
        )

    def tukey_anscombe_plot(self):
        raise NotImplementedError

    def scale_location_plot(self):
        raise NotImplementedError


@dataclass
class NormalityChecksOutput:
    targets: List[str]
    figs_qqplots: Dict[str, go.Figure]
    tables_shapiro_wilk: Dict[str, pd.DataFrame]


@dataclass
class MonitoringOutput:
    sample_size_table: pd.DataFrame
    fig_sample_size_pie: go.Figure
    fig_dynamic_sample_size: Union[go.Figure, None]
    fig_target_metric_distribution: Dict[str, go.Figure]
    fig_target_cdf: Dict[str, go.Figure]
    fig_target_qq_variants: Dict[str, go.Figure]


@dataclass
class SegmentMonitoring(Monitoring):
    segments: List[str]

    def __post_init__(self):
        assert len(self.segments) > 0
        self.count_per_segment = {
            segment: self.ed.data.groupby(segment).count()[self.ed.variant]
            for segment in self.segments
        }
        N_SEGMENTS_TO_DISPLAY = 10
        self.biggest_segments = {
            segment: self.count_per_segment[segment]
            .nlargest(N_SEGMENTS_TO_DISPLAY)
            .index.tolist()
            for segment in self.segments
        }

    def dynamic_sample_size_descriptives(self, segment, most_rlvnt_segments_only=True):
        """
        Computes the dynamic sample size descriptives for a given segment.

        Args:
            segment (str): The name of the segment to compute the descriptives for.

        Returns:
            pandas.DataFrame: A DataFrame containing the dynamic sample size descriptives for the given segment.
        """
        assert self.ed.is_dynamic_observation
        df_dyn_avg = super()._dynamic_sample_size_descriptives()
        df_base = copy.deepcopy(self.ed.data).sort_values(
            self.ed.date, ignore_index=True
        )
        cols_relevant = [self.ed.variant, self.ed.date, segment]
        df_dyn_avg = df_base[cols_relevant]

        df_dyn_avg["variant_cnt"] = df_base.groupby(
            [self.ed.variant, segment],
            group_keys=False,
        )[self.ed.date].apply(lambda x: x.expanding().count())
        if most_rlvnt_segments_only:
            df_dyn_avg = df_dyn_avg[
                df_dyn_avg[segment].isin(self.biggest_segments[segment])
            ]
        return df_dyn_avg

    def chi_squared_test_table(self, alpha=0.05):
        """
        chi-squared test for independence between the variant and each segment in the experiment.

        Args:
            alpha (float): The significance level for the test. Default is 0.05.

        Returns:
            pandas.DataFrame: A table with the p-value and decision for each segment, indicating whether the null hypothesis
            of independence between the variant and the segment can be rejected at the given significance level.
        """

        chi_squared = {"segment": [], "p-value": []}
        for _, segment in enumerate(self.segments):
            contingency_table = pd.crosstab(
                self.ed.data[self.ed.variant], self.ed.data[segment]
            )
            _, p_value, _, _ = chi2_contingency(contingency_table)
            chi_squared["segment"].append(segment)
            chi_squared["p-value"].append(p_value)
            df_chi_squared = pd.DataFrame.from_dict(chi_squared)
            is_sig_col = f"is significant at the {alpha} level"
            df_chi_squared[is_sig_col] = False
            df_chi_squared["decision"] = (
                "There is sufficient indication that there is no dependence between the"
                " segment and variant assignments."
            )
            df_chi_squared.loc[df_chi_squared["p-value"] < alpha, is_sig_col] = True
            df_chi_squared.loc[df_chi_squared["p-value"] < alpha, "decision"] = (
                "Dependence between group number and segment value (The segment is"
                " distributed disproportionally along the groups)"
            )
        return df_chi_squared

    def _chi_squared_table(self, dimension: str):
        """Create a chi-squared table for a given dimension versus variant assignment
        Creates a table with statistic (Observed frequency - Expected frequency)^2 / Expected frequency
        for each variant and dimension value combination.

        Args:
            dimension (str): segment to monitor / run independence test on

        Returns:
            pd.DataFrame: Table with chi-squared statistic for each variant and dimension value combination
        """
        ct = pd.crosstab(self.ed.data[self.ed.variant], self.ed.data[dimension])
        chi_squared_table = (ct - expected_freq(ct)) ** 2 / expected_freq(ct)
        return chi_squared_table

    def chi_squared_heatmaps(self):
        """
        Returns a dictionary of plotly figures, one for each segment, illustrating the chi-squared statistic
        """
        chi_squared_heatmaps = {}
        for j, s in enumerate(self.segments):
            chi_squared_selection = self._chi_squared_table(s).loc[
                :, self.biggest_segments[s]
            ]
            heatmap = fig_variant_segment_dependence(chi_squared_selection, self.ed)
            chi_squared_heatmaps[s] = heatmap
        return chi_squared_heatmaps

    def create_tables_and_plots(self, chi_squared_alpha=0.05):
        chi_squared_heatmaps = self.chi_squared_heatmaps()
        chi_squared_test_table = self.chi_squared_test_table(alpha=chi_squared_alpha)
        if self.ed.is_dynamic_observation:
            figs_segment_sample_size = {
                segment: plot_segment_sample_size(
                    self.ed, self.dynamic_sample_size_descriptives(segment), segment
                )
                for segment in self.segments
            }
        else:
            figs_segment_sample_size = None
        figs_segment_histograms = {
            segment: plot_segment_histograms(
                self.ed, segment, biggest_segments=self.biggest_segments[segment]
            )
            for segment in self.segments
        }

        return SegmentMonitoringOutput(
            figs_chi_squared_heatmaps=chi_squared_heatmaps,
            table_chi_squared_test=chi_squared_test_table,
            figs_segment_sample_size=figs_segment_sample_size,
            figs_segment_histograms=figs_segment_histograms,
        )


@dataclass
class SegmentMonitoringOutput:
    figs_chi_squared_heatmaps: Dict[str, go.Figure]
    table_chi_squared_test: pd.DataFrame
    figs_segment_sample_size: Union[Dict[str, go.Figure], None]
    figs_segment_histograms: Dict[str, go.Figure]


@dataclass
class SequentialTest:
    ed: ExperimentDataset

    def __post_init__(self):
        assert self.ed.is_dynamic_observation

    def sequential_tests(
        self,
        metrics: List[str],
        effect_size_means: List[float],
        sds: List[float],
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Runs sequential statistical tests on the data.

        Args:
            metrics (List[str]): A list of metric names to test.
            effect_size_means (List[float]): A list of expected effect size means for each metric.
            sds (List[float]): A list of expected standard deviations for each metric.
            alpha (float): The significance level for the tests.

        Returns:
            A pandas DataFrame containing the results of the tests.
        """
        # TODO: Create results class for sequential test

        assert len(metrics) == len(effect_size_means) == len(sds)

        df_base = copy.deepcopy(self.ed.data).sort_values(
            self.ed.date, ignore_index=True
        )

        df_dyn_avg = df_base[metrics + [self.ed.variant] + [self.ed.date]]

        df_dyn_avg[[f"{m}_avg" for m in metrics]] = df_base.groupby(
            self.ed.variant, group_keys=False
        )[metrics].apply(
            lambda x: x.expanding().mean(
                # engine="numba", engine_kwargs={"parallel": True}, raw=True
            )
        )

        df_dyn_avg[f"variant_cnt"] = df_base.groupby(self.ed.variant, group_keys=False)[
            metrics[0]
        ].apply(
            lambda x: x.expanding(
                # engine="numba", engine_kwargs={"parallel": True}
            ).count()
        )

        df_dyn_avg["is_control"] = np.where(df_dyn_avg[self.ed.variant] == 0, 1, 0)
        df_dyn_avg["control_cnt"] = (
            df_dyn_avg["is_control"]
            .expanding()
            .sum(engine="numba", engine_kwargs={"parallel": True})
        )
        # df_dyn_avg[[f"{m}_control_avg" for m in metrics]] = df_dyn_avg
        df_dyn_avg[[f"{m}_control" for m in metrics]] = df_dyn_avg[metrics].mul(
            df_dyn_avg["is_control"], axis=0
        )

        df_dyn_avg[[f"{m}_control_sum" for m in metrics]] = (
            df_dyn_avg[[f"{m}_control" for m in metrics]]
            .expanding()
            .sum(engine="numba", engine_kwargs={"parallel": True})
        )
        # rewrite the following loop to be more efficient
        for m in metrics:
            df_dyn_avg[f"{m}_control_avg"] = (
                df_dyn_avg[f"{m}_control_sum"] / df_dyn_avg["control_cnt"]
            )

        for m in metrics:
            df_dyn_avg[f"{m}_diff"] = (
                df_dyn_avg[f"{m}_avg"] - df_dyn_avg[f"{m}_control_avg"]
            )

        df_dyn_avg["avg_cnt"] = df_dyn_avg[["control_cnt", "variant_cnt"]].min(axis=1)
        # (
        # df_dyn_avg["control_cnt"] + df_dyn_avg["variant_cnt"]
        # ) / 2

        for j in range(len(metrics)):
            (
                df_dyn_avg[f"{metrics[j]}_CI_lower_no_recursion"],
                df_dyn_avg[f"{metrics[j]}_CI_upper_no_recursion"],
            ) = zip(
                *df_dyn_avg.apply(
                    lambda x: self.conf_interval_before_recursion(
                        x[f"{metrics[j]}_avg"] - x[f"{metrics[j]}_control_avg"],
                        x["avg_cnt"],
                        effect_size_means[j],
                        sds[j],
                        alpha,
                    ),
                    axis=1,
                )
            )
        try:
            df_dyn_avg = df_dyn_avg.groupby(self.ed.variant).tail(-50)
        except Exception as e:
            print("Should have at least 50 data points per variant.")

        df_dyn_avg[[f"{m}_CI_lower" for m in metrics]] = df_dyn_avg.groupby(
            self.ed.variant, group_keys=False
        )[[f"{m}_CI_lower_no_recursion" for m in metrics]].apply(
            lambda x: x.expanding().max(
                engine="numba", engine_kwargs={"parallel": True}
            )
        )

        df_dyn_avg[[f"{m}_CI_upper" for m in metrics]] = df_dyn_avg.groupby(
            self.ed.variant, group_keys=False
        )[[f"{m}_CI_upper_no_recursion" for m in metrics]].apply(
            lambda x: x.expanding().min(
                engine="numba", engine_kwargs={"parallel": True}
            )
        )

        for j in range(len(metrics)):
            df_dyn_avg[f"{metrics[j]}_inv_test_stat"] = df_dyn_avg.apply(
                lambda x: 1
                / self.msprt_test_stat(
                    x[f"{metrics[j]}_avg"] - x[f"{metrics[j]}_control_avg"],
                    x["avg_cnt"],
                    effect_size_means[j],
                    sds[j],
                ),
                axis=1,
            )

        df_dyn_avg[[f"{m}_p_values" for m in metrics]] = df_dyn_avg.groupby(
            self.ed.variant, group_keys=False
        )[[f"{m}_inv_test_stat" for m in metrics]].apply(
            lambda x: x.expanding().min(
                engine="numba",
                engine_kwargs={"parallel": True},
                # raw=True
            )
        )
        return df_dyn_avg

    def sequential_test_results(
        self,
        metrics: List = [],
        effect_size_means: dict = None,
        sds: dict = None,
        alpha: float = 0.05,
    ):
        """
        Computes sequential testing results for a list of metrics, using the effect size means and standard deviations provided, or default values if not specified.

        Args:
            metrics (List): A list of metric names to compute sequential testing results for.
            effect_size_means (dict): A dictionary mapping metric names to effect size means.
                If not provided, default values are computed based on the data (not recommended).
            sds (dict): A dictionary mapping metric names to standard deviations.
                If not provided, default values are computed based on the data (not recommended).
            alpha (float): The significance level for the sequential tests. Default is 0.05.

        Returns:
            pandas.DataFrame: A DataFrame containing the sequential testing results for each metric.
        """

        metrics = self.ed.targets
        if sds is None:
            sds = {}
            for metric in metrics:
                sds[metric] = self.ed.data.loc[
                    self.ed.data[self.ed.variant] == 0, metric
                ].std()

        DEFAULT_COHEN_D = 0.05
        if effect_size_means is None:
            effect_size_means = {}
            for metric in metrics:
                effect_size_means[metric] = (
                    self.ed.data[metric].mean() * DEFAULT_COHEN_D
                )

        df_dyn_avg = self.sequential_tests(
            metrics=list(metrics),
            effect_size_means=list(effect_size_means.values()),
            sds=list(sds.values()),
            alpha=alpha,
        )
        self._df_dyn_avg = df_dyn_avg
        return df_dyn_avg

    def msprt_test_stat(self, delta, n, tau, sigma, theta_0=0):
        norm_constant = np.sqrt(2 * sigma**2 / (2 * sigma**2 + n * tau**2))
        test_stat = norm_constant * np.exp(
            n**2
            * tau**2
            * (delta - theta_0) ** 2
            / (4 * sigma**2 * (2 * sigma**2 + n * tau**2))
        )
        if test_stat is np.nan or test_stat < 1:
            test_stat = 1
        return test_stat

    def conf_interval_before_recursion(self, delta, n, tau, sigma, alpha):
        norm_constant = np.sqrt(2 * sigma**2 / (2 * sigma**2 + n * tau**2))
        shift = (
            (-np.log(alpha * norm_constant))
            * (4 * sigma**2 * (2 * sigma**2 + n * tau**2))
            / (n**2 * tau**2)
        )
        return delta - np.sqrt(shift), delta + np.sqrt(shift)

    def fig_sequential_test(self):
        """
        Generates a plotly figure with three columns for each target metric, showing the average value over time,
        the treatment effect compared to the control group, and the p-value of the sequential test for each variant.
        The figure has one row per target metric, and each row shows the data for all variants.

        Returns:
            fig (plotly.graph_objs.Figure): the plotly figure object.
        """

        assert self._df_dyn_avg is not None, "Run sequential_test_results first"
        df_dyn_avg = self._df_dyn_avg
        fig = plot_sequential_test(self.ed, df_dyn_avg)

        return fig
