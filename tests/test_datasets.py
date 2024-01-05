import pytest

from tw_experimentation.utils import ExperimentDataset
from tw_experimentation.data_generation import RevenueConversion


@pytest.fixture(params=[True, False])
def experiment_data(request):
    rc = RevenueConversion()
    df = rc.generate_data(
        is_dynamic_assignment=request.param,
    )

    targets = ["conversion", "revenue", "num_actions"]
    metrics = ["binary", "continuous", "discrete"]

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
def boolean_binary_metric_data(experiment_data):
    ed = experiment_data
    ed.data["conversion"] = ed.data["conversion"].astype(bool)
    return ed


def test_boolean_binary_metric(boolean_binary_metric_data):
    """checks if dataset with boolean binary metric can be preprocessed"""
    ed = boolean_binary_metric_data
    assert ed.data["conversion"].dtype == bool
    ed.preprocess_dataset()


def test_check_preprocessor(experiment_data):
    """checks if dataset can be preprocessed (dummy encoding of vars etc...)

    Args:
        df (pd.DataFrame): dataset for experiments, with cols for treatment, targets and covariates
    """
    ed = experiment_data
    ed.preprocess_dataset()


if __name__ == "__main__":
    pytest.main([__file__])
