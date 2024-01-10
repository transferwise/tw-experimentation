import pytest

import pandas as pd
import numpy as np
from tw_experimentation.utils import ExperimentDataset
from tw_experimentation.data_generation import RevenueConversion

from tw_experimentation.statistical_tests import FrequentistTest, BaseTest

from tw_experimentation.widgetizer import Monitoring, MonitoringInterface
from tw_experimentation.checker import (
    Monitoring,
    SegmentMonitoring,
    SequentialTest,
    NormalityChecks,
)
import plotly.graph_objects as go


@pytest.fixture(params=[True, False])
def experiment_data_with_segments(request):
    np.random.seed(40)

    rc = RevenueConversion()

    df_abn = rc.generate_data_abn_test(
        n=10000,
        n_treatments=2,
        baseline_conversion=0.4,
        treatment_effect_conversion=0.04,
        baseline_mean_revenue=6,
        sigma_revenue=1,
        treatment_effect_revenue=0.1,
    )

    df_abn["currency"] = np.random.choice(["GBP", "EUR", "USD"], size=len(df_abn))
    df_abn["country_of_origin"] = np.random.choice(
        ["UK", "US", "USD"], size=len(df_abn)
    )
    df_abn["segment_1"] = np.random.choice(["New", "Old"], size=(len(df_abn), 1))
    df_abn["segment_2"] = np.random.choice(
        ["Active", "Rare", "Usual"], size=(len(df_abn), 1)
    )
    df_abn["segment_3"] = np.random.choice(
        ["10+ transfers", "10- transfers"], size=(len(df_abn), 1)
    )
    is_dynamic_assignment = request.param
    ed = ExperimentDataset(
        data=df_abn,
        variant="T",
        targets=["conversion", "revenue"],
        date="trigger_dates" if is_dynamic_assignment else None,
        n_variants=2,
    )
    if not is_dynamic_assignment:
        assert not ed.is_dynamic_observation
    ed.preprocess_dataset(remove_outliers=True)
    return ed


def test_monitoring_segments_in_notebook(experiment_data_with_segments):
    ed = experiment_data_with_segments

    # choose segments to monitor
    segments = ["currency", "segment_1", "segment_2"]
    segment_monitor = SegmentMonitoring(ed, segments)
    segment_monitor_results = segment_monitor.create_tables_and_plots()


def test_monitoring_in_notebook(experiment_data_with_segments):
    ed = experiment_data_with_segments
    monitor = Monitoring(ed)
    monitor_results = monitor.create_tables_and_plots()


def test_normality_checks(experiment_data_with_segments):
    ALPHA = 0.01
    ed = experiment_data_with_segments
    nc = NormalityChecks(ed)
    nco = nc.create_results(alpha=ALPHA)
    assert len(nco.figs_qqplots) > 0
    assert len(nco.tables_shapiro_wilk) > 0
    assert isinstance(nco.figs_qqplots["revenue"], go.Figure)


def test_sequential_test_in_notebook(experiment_data_with_segments):
    ed = experiment_data_with_segments
    if ed.is_dynamic_observation:
        sequential_test = SequentialTest(ed)
        sequential_test_results = sequential_test.sequential_test_results()
        fig = sequential_test.fig_sequential_test()


if __name__ == "__main__":
    pytest.main([__file__])
