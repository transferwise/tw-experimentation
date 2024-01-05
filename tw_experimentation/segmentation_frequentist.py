from tw_experimentation.utils import ExperimentDataset, highlight
from tw_experimentation.statistical_tests import (
    FrequentistTest,
    FrequentistTestResults,
    compute_frequentist_results,
)
from typing import List, Optional, Dict, Union
from dataclasses import dataclass

import warnings
import copy

import pandas as pd
import json

from wise_pizza import explain_levels
from wise_pizza.plotting import plot_segments
import plotly.graph_objects as go
from IPython.display import display


@dataclass
class SegmentationOutput:
    """
    Represents the output of a segmentation analysis.

    Attributes:
        segments (List[str]): List of segment names.
        wise_pizza (pd.DataFrame): DataFrame containing the results of the analysis for each segment.
        segment_results (Dict[str, Dict[str, FrequentistTestResults]]):
            Dictionary containing the results of the frequentist tests for each segment.
            segment_results[category][segment] contains for each category the frequentist output for the segment.

    Methods:
        segment_output(segment: str) -> pd.DataFrame:
            Returns the results table for a specific segment.
    """

    segments: List[str]
    wise_pizza_figs: Dict[str, Dict[str, go.Figure]]
    wise_pizza_results: Dict[str, Dict[str, pd.DataFrame]]
    segment_results: Dict[str, Dict[str, FrequentistTestResults]]

    def segment_output(
        self, category: str, multitest_correction="bonferroni", alpha=0.05
    ) -> pd.DataFrame:
        """
        Returns the results table for a specific segment.

        Args:
            segment (str): The name of the segment.

        Returns:
            pd.DataFrame: The results table for the specified segment.
        """
        segment_tables = {
            k: self.segment_results[category][k]
            .compute_stats_per_target(
                multitest_correction=multitest_correction, type_i_error=alpha
            )
            .get_results_table()
            for k in self.segment_results[category]
        }
        results_table = pd.concat(
            segment_tables.values(), axis=0, keys=segment_tables.keys()
        )
        results_table.index.names = [category] + segment_tables[
            list(segment_tables.keys())[0]
        ].index.names[:]

        results_table = results_table.swaplevel(0, 2).sort_index()
        return results_table

    def wise_pizza_output(
        self,
        target: str,
        treatment: int,
        auto_display_df=False,
        multitest_correction="bonferroni",
    ):
        wise_pizza_df = {
            k: self.wise_pizza_results[target][treatment][k]
            .compute_stats_per_target(multitest_correction=multitest_correction)
            .get_results_table()
            for k in self.wise_pizza_results[target][treatment].keys()
        }
        results_table = pd.concat(
            wise_pizza_df.values(),
            axis=0,
            keys=wise_pizza_df.keys(),
        )
        results_table.index.names = ["Pizza slices", "Variants", "Outcomes"]
        results_table = results_table.swaplevel(0, 2)
        results_table = results_table.sort_index()
        if auto_display_df:
            display(
                results_table.style.apply(highlight, axis=1)
                .set_table_attributes("style='display:inline'")
                .bar(subset=["Estimated_Effect_relative"], color="grey")
                .format(precision=3)
            )
        return results_table


def run_segmentation_analysis(ed: ExperimentDataset, segments: Union[None, List[str]]):
    if segments is None:
        segments = ed.segments
    assert len(segments) > 0, "No segments specified"
    assert all([s in ed.data.columns for s in segments]), "Segment not in data"
    WP_MIN_SEGMENTS = 2
    WP_MAX_DEPTH = 3

    segmentation = SegmentationNew(ed)
    segment_outputs = {k: segmentation.frequentist_by_segment(k) for k in segments}
    wise_pizza_frequentist = dict()
    wise_pizza_figs = dict()
    for target in ed.targets:
        wise_pizza_figs[target] = dict()
        wise_pizza_frequentist[target] = dict()
        for treatment in range(1, ed.n_variants):
            (
                wise_pizza_frequentist[target][treatment],
                wise_pizza_figs[target][treatment],
            ) = segmentation.wise_pizza_frequentist(
                target=target,
                treatment=treatment,
                segments=segments,
                max_depth=WP_MAX_DEPTH,
                min_segments=WP_MIN_SEGMENTS,
            )

    so = SegmentationOutput(
        segments=segments,
        wise_pizza_figs=wise_pizza_figs,
        wise_pizza_results=wise_pizza_frequentist,
        segment_results=segment_outputs,
    )

    return so


class SegmentationNew:
    def __init__(self, ed) -> None:
        self.ed = ed
        self.df_wp = None
        self.sf = None

    def frequentist_by_segment(self, category: str, max_segments: int = 7):
        assert category in self.ed.data.columns

        segments = (
            self.ed.data.groupby(category)
            .count()
            .sort_values(by=[self.ed.variant], ascending=False)
            .index.values.tolist()[:max_segments]
        )
        segment_results = {}
        for segment in segments:
            ed_segment = copy.deepcopy(self.ed)
            ed_segment.data = ed_segment.data[ed_segment.data[category] == segment]
            segment_results[segment] = compute_frequentist_results(ed_segment)
        return segment_results

    def wise_pizza_frequentist(
        self,
        target: str,
        treatment: int = 1,
        segments=List[str],
        max_depth: int = 3,
        min_segments: int = 2,
    ):
        """Fit method for wise pizza to get wise pizza results and frequentist analysis of segments

        Args:
            treatment (int, optional): name of variant at consideration. Defaults to 1.
            segments (List, optional): categorical segments to consider. Defaults to List[str].
            max_depth (int, optional): wise pizza parameter for max number of segment combinations. Defaults to 2.
            min_segments (int, optional): wise pizza parameter for minimal no. of segments to display. Defaults to 3.

        Returns:
            pd.DataFrame: frequentist result table for wise pizza selected segments
        """
        # prepare the dough (i.e. dataframe) of a wise pizza
        df_wp = self._wise_pizza_prep(treatment=treatment, segments=segments)

        # run Wise Pizza
        sf = explain_levels(
            df=df_wp,
            dims=segments,
            total_name=target,
            size_name=self.ed.variant,
            max_depth=max_depth,
            min_segments=min_segments,
        )
        self.sf = sf
        wp_fig = plot_segments(sf, width=300, height=300, return_fig=True)
        wp_segments = [s["segment"] for s in sf.segments]

        segment_results = dict()
        for seg in wp_segments:
            query_rows = pd.Series(True, index=range(len(self.ed.data)))
            for k, v in seg.items():
                query_rows = (query_rows) & (self.ed.data[k] == v)
            ed_segment = copy.deepcopy(self.ed)
            ed_segment.data = self.ed.data[query_rows]
            ed_segment.n_variants = 2
            segment_results[json.dumps(seg)] = compute_frequentist_results(ed_segment)

        return segment_results, wp_fig

    def _wise_pizza_prep(self, treatment: int = 1, segments: [] = List[str]):
        ps_counts = (
            self.ed.data.loc[self.ed.data[self.ed.variant].isin([0, treatment]), :]
            .groupby(by=segments)[self.ed.variant]
            .agg("count")
        )

        df_wp_prep = (
            self.ed.data.loc[self.ed.data[self.ed.variant].isin([0, treatment]), :]
            .groupby(by=segments + [self.ed.variant])[list(self.ed.targets)]
            .agg(["mean"])
        )

        # df_wp_prep[self.ed.variant] = df_wp_prep.join(ps_counts, on=segments)

        df_test = df_wp_prep.query(f"{self.ed.variant} == {treatment}").reset_index(
            level=[self.ed.variant]
        )
        df_ctrl = df_wp_prep.query(f"{self.ed.variant} == 0").reset_index(
            level=[self.ed.variant]
        )

        df_combined = self._te_per_segment_naive(df_test, df_ctrl)

        df_wp = df_combined.merge(
            ps_counts,
            left_index=True,
            right_index=True,
        )
        df_wp.reset_index(inplace=True)

        for m in self.ed.targets:
            df_wp[m] = df_wp[m] * df_wp[self.ed.variant]

        self.df_wp = df_wp

        return df_wp

    def get_sf(self):
        assert (
            self.sf is not None
        ), "You need to run wise_pizza_frequentist (fit) first!"
        return self.sf

    def _te_per_segment_naive(self, df_test, df_ctrl):
        # to be replaced by causaltune
        df_combined = df_test.merge(
            df_ctrl, left_index=True, right_index=True, suffixes=("_treat", "_ctrl")
        )

        for m in self.ed.targets:
            df_combined[m] = df_combined[f"{m}_treat"] - df_combined[f"{m}_ctrl"]
        df_combined = df_combined[list(self.ed.targets)]
        df_combined.columns = df_combined.columns.droplevel(level=1)

        return df_combined


class Segmentation:
    def __init__(self, ed, alpha: Optional[float] = 0.05) -> None:
        self.ed = ed
        self.df_wp = None
        self.sf = None
        self.alpha = alpha

    def frequentist_by_segment(self, category: str, max_segments: int = 7):
        assert category in self.ed.data.columns

        segments = (
            self.ed.data.groupby(category)
            .count()
            .sort_values(by=[self.ed.variant], ascending=False)
            .index.values.tolist()[:max_segments]
        )
        segment_results = {}
        for segment in segments:
            ed_segment = copy.deepcopy(self.ed)
            ed_segment.data = ed_segment.data[ed_segment.data[category] == segment]
            ft_segment = FrequentistTest(ed=ed_segment, alpha=self.alpha)
            ft_segment.compute()
            segment_results[segment] = ft_segment.get_results_table()

        results_table = pd.concat(
            segment_results.values(), axis=0, keys=segment_results.keys()
        )
        results_table.index.names = [category] + segment_results[
            segments[0]
        ].index.names[:]

        results_table = results_table.swaplevel(0, 2).sort_index()

        warnings.warn(
            "There is no p-value correction for segmentation analsysis implemented at"
            " this point."
        )

        return results_table

    def wise_pizza_frequentist(
        self,
        target: str,
        treatment: int = 1,
        segments=List[str],
        max_depth: int = 2,
        min_segments: int = 3,
        auto_display_df=True,
    ):
        """Fit method for wise pizza to get wise pizza results and frequentist analysis of segments

        Args:
            treatment (int, optional): name of variant at consideration. Defaults to 1.
            segments (List, optional): categorical segments to consider. Defaults to List[str].
            max_depth (int, optional): wise pizza parameter for max number of segment combinations. Defaults to 2.
            min_segments (int, optional): wise pizza parameter for minimal no. of segments to display. Defaults to 3.

        Returns:
            pd.DataFrame: frequentist result table for wise pizza selected segments
        """
        # prepare the dough (i.e. dataframe) of a wise pizza
        df_wp = self._wise_pizza_prep(treatment=treatment, segments=segments)

        # run Wise Pizza
        sf = explain_levels(
            df=df_wp,
            dims=segments,
            total_name=target,
            size_name=self.ed.variant,
            max_depth=max_depth,
            min_segments=min_segments,
        )
        self.sf = sf

        wp_segments = [s["segment"] for s in sf.segments]

        segment_results = {}
        for seg in wp_segments:
            query_rows = pd.Series(True, index=range(len(self.ed.data)))
            for k, v in seg.items():
                query_rows = (query_rows) & (self.ed.data[k] == v)
            ed_segment = copy.deepcopy(self.ed)
            ed_segment.data = self.ed.data[query_rows]
            ed_segment.n_variants = 2
            ft_segment = FrequentistTest(ed=ed_segment, alpha=self.alpha)
            ft_segment.compute()
            segment_results[json.dumps(seg)] = ft_segment.get_results_table()
            results_table = pd.concat(
                segment_results.values(), axis=0, keys=segment_results.keys()
            )

        results_table.index.names = ["Pizza slices", "Variants", "Outcomes"]
        results_table = results_table.swaplevel(0, 2)
        results_table = results_table.sort_index()
        if auto_display_df:
            display(
                results_table.style.apply(highlight, axis=1)
                .set_table_attributes("style='display:inline'")
                .bar(subset=["Estimated_Effect_relative"], color="grey")
                .format(precision=3)
            )
        return results_table

    def _wise_pizza_prep(self, treatment: int = 1, segments=List[str]):
        ps_counts = (
            self.ed.data.loc[self.ed.data[self.ed.variant].isin([0, treatment]), :]
            .groupby(by=segments)[self.ed.variant]
            .agg("count")
        )

        df_wp_prep = (
            self.ed.data.loc[self.ed.data[self.ed.variant].isin([0, treatment]), :]
            .groupby(by=segments + [self.ed.variant])[list(self.ed.targets)]
            .agg(["mean"])
        )

        # df_wp_prep[self.ed.variant] = df_wp_prep.join(ps_counts, on=segments)

        df_test = df_wp_prep.query(f"{self.ed.variant} == {treatment}").reset_index(
            level=[self.ed.variant]
        )
        df_ctrl = df_wp_prep.query(f"{self.ed.variant} == 0").reset_index(
            level=[self.ed.variant]
        )

        df_combined = self._te_per_segment_naive(df_test, df_ctrl)

        df_wp = df_combined.merge(
            ps_counts,
            left_index=True,
            right_index=True,
        )
        df_wp.reset_index(inplace=True)

        for m in self.ed.targets:
            df_wp[m] = df_wp[m] * df_wp[self.ed.variant]

        self.df_wp = df_wp

        return df_wp

    def get_sf(self):
        assert (
            self.sf is not None
        ), "You need to run wise_pizza_frequentist (fit) first!"
        return self.sf

    def _te_per_segment_naive(self, df_test, df_ctrl):
        # to be replaced by autocausality
        df_combined = df_test.merge(
            df_ctrl, left_index=True, right_index=True, suffixes=("_treat", "_ctrl")
        )

        for m in self.ed.targets:
            df_combined[m] = df_combined[f"{m}_treat"] - df_combined[f"{m}_ctrl"]
        df_combined = df_combined[list(self.ed.targets)]
        df_combined.columns = df_combined.columns.droplevel(level=1)

        return df_combined
