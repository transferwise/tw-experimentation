import pytest

import pandas as pd
import numpy as np
from tw_experimentation.utils import ExperimentDataset
from tw_experimentation.data_generation import RevenueConversion

from tw_experimentation.statistical_tests import FrequentistTest, BaseTest
from tw_experimentation.segmentation_frequentist import Segmentation

from tw_experimentation.widgetizer import (
    FrequentistEvaluation,
)

from tw_experimentation.statistical_tests import cuped


@pytest.fixture
def experiment_data():
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


@pytest.fixture
def experiment_data_two_treatment():
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

    ed = ExperimentDataset(
        data=df_abn,
        variant="T",
        targets=["conversion", "revenue"],
        date="trigger_dates",
        pre_experiment_cols=["pre_exp_revenue"],
        n_variants=2,
    )
    ed.preprocess_dataset()
    return ed


@pytest.fixture
def experiment_data_with_segments():
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
    ed = ExperimentDataset(
        data=df_abn,
        variant="T",
        targets=["conversion", "revenue"],
        date="trigger_dates",
        n_variants=2,
    )
    ed.preprocess_dataset(remove_outliers=True)
    return ed


@pytest.fixture
def no_effect_experiment_data():
    N = 1000
    df = pd.DataFrame(
        data={
            "variant": np.random.choice([0, 1], size=(N,)),
            "outcome": 100 * np.ones(N),
        }
    )
    ed = ExperimentDataset(
        data=df,
        variant="variant",
        targets=["outcome"],
        date=None,
        metric_types={"outcome": "continuous"},
    )
    ed.preprocess_dataset()
    return ed


def test_frequentist(experiment_data):
    ed = experiment_data
    ed.preprocess_dataset()
    ft = FrequentistTest(ed=ed)
    ft.compute()
    ft.get_results_table()


def test_naive_no_effect_base_estimate(no_effect_experiment_data):
    ed = no_effect_experiment_data
    ed.preprocess_dataset()
    bt = BaseTest(ed=ed)
    assert bt.naive_relative_treatment_effects["outcome"][1] == 0


def test_frequentist_in_notebook(experiment_data_two_treatment):
    evaluation = FrequentistEvaluation(experiment_data_two_treatment)
    evaluation.start()

    cuped(experiment_data_two_treatment, has_correction="Yes", alpha=0.05)


# def test_frequentist_wise_pizza_segmentation_in_notebook(experiment_data_with_segments):
#     segments = ["segment_1", "segment_2", "segment_3", "currency"]

#     # change to treatment number of interest
#     treatment = 1

#     segmentation = Segmentation(ed=experiment_data_with_segments)
#     segmentation.wise_pizza_frequentist(
#         target="conversion",
#         treatment=treatment,
#         segments=segments,
#         max_depth=2,  # customize to own preference
#         min_segments=10,  # customize to own preference
#     )

#     slice_finder = segmentation.get_sf()
#     slice_finder.plot(width=1000, plot_is_static=True)


if __name__ == "__main__":
    pytest.main([__file__])
