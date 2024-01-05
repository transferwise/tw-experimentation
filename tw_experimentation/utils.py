import os
from dataclasses import dataclass
from typing import List, Union, Optional
import pandas as pd
import numpy as np
from numpy.distutils.misc_util import is_sequence
from itertools import repeat
from sklearn.utils import resample

from tw_experimentation.constants import PLOTLY_COLOR_PALETTE, MetricType


def highlight(df):
    """Highlight significant results in green, non-significant in red for frequentist stat table"""
    if df["is_significant"]:
        if df["Estimated_Effect_relative"] > 0:
            return ["background-color: lightgreen"] * len(df)
        else:
            return ["background-color: lightcoral"] * len(df)
    else:
        return [""] * len(df)


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color format to rgb
    Taken from
    https://community.plotly.com/t/scatter-plot-fill-with-color-how-to-set-opacity-of-fill/29591/2
    Args:
        hex_color (str): hex color in format '#rrggbb'

    Returns:
        tuple: rgb color as tuple
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def variant_name_map(n_variants: int):
    """Name variants to be used in plots

    Args:
        n_variants (int): Number of variants, includes control

    Returns:
        dict: map from variant number to variant name
    """
    assert n_variants > 0
    names = [
        f"Variant {i}" + (" (treatment)" if i > 0 else " (control)")
        for i in range(n_variants)
    ]

    name_map = dict(zip(range(n_variants), names))
    return name_map


def variant_color_map(n_variants: int):
    assert n_variants > 0
    color_map = dict(zip(range(n_variants), PLOTLY_COLOR_PALETTE))
    return color_map


def variantname_color_map(variant_names: List[str]):
    # """Map variant names to colors for plotting"""
    assert len(variant_names) > 0
    color_map = dict(zip(variant_names, PLOTLY_COLOR_PALETTE))
    return color_map


def create_dir(loc: str):
    parent = os.path.realpath(os.path.join(loc, ".."))
    if os.path.isdir(parent):
        if not os.path.isdir(loc):
            print("creating", loc)
            os.mkdir(loc)
    else:
        create_dir(parent)
        create_dir(loc)


@dataclass
class ExperimentMetaData:
    variant: str
    targets: List[str]
    n_variants: int
    date: Optional[str] = None
    pre_experiment_cols: Optional[List[str]] = None
    segments: Optional[List[str]] = None
    metric_types: Optional[dict[str, str]] = None
    is_dynamic_observation: Optional[bool] = None
    is_only_pre_experiment: Optional[bool] = False
    variant_labels: Optional[List[str]] = None
    variant_names: Optional[List[str]] = None
    sample_sizes: Optional[dict[str, int]] = None
    total_sample_size: Optional[int] = None
    target_standard_deviations: Optional[dict[str, float]] = None


@dataclass
class ExperimentDataset:
    data: pd.DataFrame
    variant: str
    targets: List[str]
    date: str
    ratio_targets: List[tuple]  # tuples column names for ratio metric calculation
    n_variants: Optional[int]

    def __init__(
        self,
        data: pd.DataFrame,
        variant: str,
        targets: Union[str, List[str]],
        pre_experiment_cols: Optional[List[str]] = None,
        segments: Optional[List[str]] = None,
        metric_types: Optional[dict[str, str]] = None,
        date: Optional[str] = None,
        ratio_targets: Optional[dict[str, tuple]] = None,
        n_variants: Optional[int] = 2,
        control_label: Optional[str] = 0,
        is_dynamic_observation: Union[Optional[bool], None] = None,
        is_only_pre_experiment: Optional[bool] = False,
    ) -> None:
        """
        Implements data logic for A/B testing. Assumes that observations are already on experiment analysis level.

        Args:
            data (pd.DataFrame): data with columns variant, target, date
            variant (str): variant column name
            targets (Union[str, List[str]]): target column name(s)
            pre_experiment_cols (Optional[List[str]], optional): pre-experimental data columns. Defaults to None.
            metric_types (Optional[dict[str, str]], optional): metric types ('binary', 'discrete', or 'continuous').
                Defaults to None.
            date (Optional[str], optional): timestamp column. Defaults to None.
            ratio_targets (Optional[dict[str, tuple]], optional): ratio targets with numerator and denominator in tuple.
                Not implemented yet. Defaults to None.
            n_variants (Optional[int], optional): Number of variants TODO: autodetect this. Defaults to 2.
            control_label (Optional[str], optional): Label of the control group variant. Defaults to 0.
            is_dynamic_observation (Optional[bool], optional): Whether the assignment are dynamic
                (for monitoring and sequential analysis). Defaults to True.
            is_only_pre_experiment (Optional[bool], optional): Whether it only uses pre-experimental data.
        """
        # e.g. ratio_targets {'volume_per_transaction':
        # (total_volumes_per_customer, n_transactions_per_customer)}
        # will calculated mean and variance, covariance based on these stats

        assert isinstance(data, pd.DataFrame)
        self.data = data.copy()
        self.metric_types = metric_types
        self.date = date
        self.ratio_targets = ratio_targets
        self.n_variants = n_variants
        self.segments = segments or []
        self.control_label = control_label
        self.pre_experiment_cols = pre_experiment_cols or []
        self.is_dynamic_observation = is_dynamic_observation
        self.is_only_pre_experiment = is_only_pre_experiment
        self.variant = variant

        if not self.is_only_pre_experiment:
            assert isinstance(
                variant, str
            ), "Only a single treatment column supported at the moment"
            assert is_sequence(targets)
            self.n_variants = self.data[self.variant].nunique()

        self.targets = [targets] if isinstance(targets, str) else targets

        self.is_preprocessed = False

    def preprocess_pre_experiment_dataset(self):
        """
        - detec metric types
        """
        assert (
            self.is_only_pre_experiment is True
        ), "Can only preprocess data from pre-experiment"

        self._detect_metric_types(self.pre_experiment_cols)
        self._nan_data_cleaning()
        self._binary_metric_value_check()

    def preprocess_dataset(self, remove_outliers=True):
        """
        - calculate ratio metrics
        - remove outliers (optionally)
        """
        assert (
            self.is_only_pre_experiment is False
        ), "Can only preprocess data from experiment"
        # TODO: check if variant = 0,1[,2,...] or string. If the latter, ask for name of control group
        # For now, assume that control = 0, treatment = 1,2,3,...

        # TODO: Convert date column to datetime
        self.data[self.variant] = self.data[self.variant].astype(int)
        try:
            assert 0 in list(self.data[self.variant].unique()) and 1 in list(
                self.data[self.variant].unique()
            )
        except:
            raise Exception("Variants must be 0 (=control), 1,[2,...] (=treatment)")

        experiment_cols = [
            self.targets,
            [self.variant],
            self.segments,
            self.pre_experiment_cols,
        ]
        if self.date is not None:
            experiment_cols.append([self.date])

        assert all(
            (
                set(col_type).issubset(set(self.data.columns)),
                "Columns must be in dataframe",
            )
            for col_type in experiment_cols
        )

        experiment_cols_list = [col for cols in experiment_cols for col in cols]
        assert len(set(experiment_cols_list)) == len(
            experiment_cols_list
        ), "Columns must be unique"

        if self.metric_types is None:
            self.metric_types = dict(zip(self.targets, repeat(None)))
        missing_metric_types = [
            t for t, v in self.metric_types.items() if v is None and t in self.targets
        ]
        if self.pre_experiment_cols is not None and len(self.pre_experiment_cols) > 0:
            missing_metric_types += self.pre_experiment_cols

        self._nan_data_cleaning()
        self._detect_metric_types(missing_metric_types)
        available_types = [metric_type.value for metric_type in MetricType]
        assert all(
            metric_type in available_types for metric_type in self.metric_types.values()
        ), "Metric types must be one of 'binary', 'discrete', or 'continuous'"

        self._binary_metric_value_check()

        if remove_outliers:
            self._remove_outliers()

        if self.is_dynamic_observation is None:
            self._detect_is_dynamic_observation()

        self.is_preprocessed = True

    def _binary_metric_value_check(self):
        assert all(
            self.data[metric].isin([0, 1]).all()
            for metric, metric_type in self.metric_types.items()
            if metric_type == MetricType.BINARY.value
        ), "Binary metric types must consist of values 0 or 1"

    def _detect_is_dynamic_observation(self):
        """Detect whether the assignment of variants is dynamic (for monitoring and sequential analysis)"""
        if self.date is not None and self.data[self.date].nunique() > 1:
            self.is_dynamic_observation = True
        else:
            self.is_dynamic_observation = False

    def _nan_data_cleaning(self):
        self.data = self.data.fillna(0)

    def _detect_metric_types(self, metrics):
        """Detect metric types with the following logic:
            if integer:
                check whether binary 0,1
                    yes -> 'binary'
                    no -> 'discrete
            elif other type:
                try converting to float
                if binary -> 'binary'
                elif float is integer -> 'discrete'
                else -> 'continuous'
            else raise error

        Args:
            metrics (List[str]): List of metrics for which to detect metric type
                Must be columns in self.data

        Raises:
            TypeError: If metrics cannot be converted to number, raise error

        """
        assert set(metrics).issubset(self.data.columns)
        if self.metric_types is None:
            self.metric_types = {}
        for metric in metrics:
            error_msg = f"Metrics must be numbers, but column {metric} is not."
            auto_dtype = self.data[metric].dtype
            if isinstance(self.data[metric].dtype, int):
                if set(self.data[metric].values) == set([0, 1]):
                    self.metric_types[metric] = MetricType.BINARY.value
                else:
                    self.metric_types[metric] = MetricType.DISCRETE.value
            elif isinstance(auto_dtype, (float, str, object)):
                try:
                    self.data[metric] = self.data[metric].fillna(np.nan).apply(float)
                    if self.data[metric].apply(float.is_integer).all():
                        self.data[metric].fillna(np.nan).apply(int)
                        if set(self.data[metric].values) == set((0, 1)):
                            self.metric_types[metric] = MetricType.BINARY.value
                        else:
                            self.metric_types[metric] = MetricType.DISCRETE.value
                    else:
                        self.metric_types[metric] = MetricType.CONTINUOUS.value
                except Exception as TypeError:
                    print(error_msg)
            else:
                raise TypeError(error_msg)

    def _aggregator(self):
        # Similar as aggregate preprocessor in
        # https://github.com/MobileTeleSystems/Ambrosia/blob/main/examples/1_usage_example_for_core_classes.ipynb
        pass

    def _remove_outliers(self):
        outliers_dict = {}
        for target in self.targets:
            if self.metric_types[target] == MetricType.CONTINUOUS.value:
                outliers_dict[target] = np.percentile(self.data[target], 99)
        if len(outliers_dict) > 0:
            query_string = " and ".join(
                [
                    f"{target}<={perc}"
                    for (target, perc) in zip(
                        outliers_dict.keys(), outliers_dict.values()
                    )
                ]
            )
            self.data = self.data.query(query_string)

    def _diff_in_diff(self):
        # to remove pre-experimental bias
        pass

    def experiment_meta_data(self):
        emd = ExperimentMetaData(
            variant=self.variant,
            targets=self.targets,
            n_variants=self.n_variants,
            date=self.date,
            pre_experiment_cols=self.pre_experiment_cols,
            segments=self.segments,
            metric_types=self.metric_types,
            is_dynamic_observation=self.is_dynamic_observation,
            is_only_pre_experiment=self.is_only_pre_experiment,
            variant_labels=self.variant_labels,
            variant_names=self.variant_names,
            sample_sizes=self.sample_sizes,
            total_sample_size=self.total_sample_size,
            target_standard_deviations=self.target_standard_deviations,
        )
        return emd

    @property
    def target_standard_deviations(self):
        """Compute standard deviation of each target for control group"""
        return {
            target: self.data.loc[self.data[self.variant] == 0, target].std()
            for target in self.targets
        }

    @property
    def variant_labels(self):
        return np.sort(self.data[self.variant].unique())

    @property
    def get_metric_types(self):
        return self.metric_types

    @property
    def variant_names(self):
        # TODO: Check whether this is duplicated from variant_labels
        return variant_name_map(self.n_variants)

    @property
    def sample_sizes(self):
        return {
            label: len(self.data[self.data[self.variant] == label].index)
            for label in self.variant_labels
        }

    @property
    def total_sample_size(self):
        return len(self.data)
