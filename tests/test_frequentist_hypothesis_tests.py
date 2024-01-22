import pytest

from tw_experimentation.utils import ExperimentDataset
from tw_experimentation.data_generation import RevenueConversion

from tw_experimentation.statistical_tests import (
    FrequentistTest,
    compute_frequentist_results,
)

from pandas import testing as tm


@pytest.fixture
def experiment_data_continuous(request):
    rc = RevenueConversion()
    df = rc.generate_data()

    targets = ["revenue"]
    metrics = ["continuous"]

    ed = ExperimentDataset(
        data=df,
        variant="T",
        targets=targets,
        date="trigger_dates",
        metric_types=dict(zip(targets, metrics)),
    )
    ed.preprocess_dataset()
    return ed


@pytest.fixture
def experiment_data(request):
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
    return ed


@pytest.mark.parametrize("ed", ["experiment_data", "experiment_data_continuous"])
def test_welsh_t_test(ed, request):
    ed = request.getfixturevalue(ed)
    ed.preprocess_dataset()
    ft = FrequentistTest(ed=ed)
    ft.compute()
    ft_df = ft.get_results_table()

    ftr = compute_frequentist_results(ed=ed)
    ftr.compute_stats_per_target(multitest_correction="bonferroni")
    ftr_df = ftr.get_results_table()
    tm.assert_series_equal(ftr_df["Control_Group_Mean"], ft_df["Control_Group_Mean"])
    tm.assert_series_equal(
        ftr_df["Treatment_Group_Mean"], ft_df["Treatment_Group_Mean"]
    )
    tm.assert_series_equal(
        ftr_df["Estimated_Effect_absolute"], ft_df["Estimated_Effect_absolute"]
    )
    tm.assert_series_equal(
        ftr_df["Estimated_Effect_relative"], ft_df["Estimated_Effect_relative"]
    )
    tm.assert_series_equal(
        ftr_df["p_value"], ft_df["p_value"], check_exact=False, atol=1e-2
    )
    tm.assert_series_equal(
        ftr_df["CI_lower"], ft_df["CI_lower"], check_exact=False, atol=1e-2
    )
    tm.assert_series_equal(
        ftr_df["CI_upper"], ft_df["CI_upper"], check_exact=False, atol=1e-2
    )


if __name__ == "__main__":
    pytest.main([__file__])
