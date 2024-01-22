import pytest

from tw_experimentation.utils import ExperimentDataset
from tw_experimentation.data_generation import RevenueConversion
from tw_experimentation.setuper import ExpDesignAutoCalculate, Setuper
import numpy as np


@pytest.fixture
def experiment_data():
    rc = RevenueConversion()
    df = rc.generate_data()

    ed = ExperimentDataset(
        data=df,
        variant="",
        targets="",
        pre_experiment_cols=["revenue", "conversion"],
        is_only_pre_experiment=True,
    )
    ed.preprocess_pre_experiment_dataset()
    return ed


def test_check_preprocessor(experiment_data):
    """checks if dataset can be preprocessed
        (dummy encoding of vars etc...)

    Args:
        df (pd.DataFrame): dataset for experiments, with cols
            for treatment, targets and covariates
    """
    ed = experiment_data
    ed.preprocess_pre_experiment_dataset()


def test_auto_sample_size_filler(experiment_data):
    ed = experiment_data
    edac = ExpDesignAutoCalculate(ed)
    for target in ["revenue", "conversion"]:
        metric_type = edac.metric_types[target]
        if metric_type in ["continuous", "discrete"]:
            exp_design_metric_type = "continuous"
            exp_design_baseline_mean = edac.mean(target)
            exp_design_sd = edac.sd(target)
        elif metric_type == "binary":
            exp_design_metric_type = "binary"
            exp_design_baseline_conversion = edac.mean(target)

        if exp_design_metric_type == "binary":
            sd = np.sqrt(
                exp_design_baseline_conversion * (1 - exp_design_baseline_conversion)
            )
            setup = Setuper.from_uplift(
                alpha=0.05,
                beta=0.2,
                uplift=0.1,
                sd=sd,
                mean=exp_design_baseline_conversion,
                relation="relative",
                treatment_share=0.5,
            )
            _ = setup.sample_size_two_sample_proportion_z_test()
        elif exp_design_metric_type == "continuous":
            setup = Setuper.from_uplift(
                alpha=0.1,
                beta=0.2,
                uplift=0.1,
                sd=exp_design_sd,
                mean=exp_design_baseline_mean,
                relation="relative",
                treatment_share=0.4,
            )
            setup.sample_size_t_test()
        else:
            raise ValueError("metric type not supported")


if __name__ == "__main__":
    pytest.main([__file__])
