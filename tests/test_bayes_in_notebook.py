import numpy as np
import pytest

from tw_experimentation.bayes.bayes_test import BayesTest
from tw_experimentation.data_generation import RevenueConversion
from tw_experimentation.utils import ExperimentDataset


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


def test_bayestest_end(experiment_data_two_treatment):
    ed = experiment_data_two_treatment
    bt = BayesTest(ed=ed)
    br = bt.compute_posterior()

    outcome_metric = "conversion"

    br.fig_posterior_difference_by_target(outcome_metric)

    # greater than zero probability
    description = f"The probability that the average treatment effect is greater than 0 for {outcome_metric} is\n"  # noqa: E501
    for variant in range(1, ed.n_variants):
        description += f" {br.prob_greater_than_zero(outcome_metric)[variant]*100}% for variant {variant}"  # noqa: E501
        description += "." if variant == ed.n_variants - 1 else ",\n"
    print(description)

    # greater than threshold probability
    threshold = 100
    description = "The probability that the average treatment effect is greater than "
    f"{threshold} for {outcome_metric} is\n"
    for variant in range(1, ed.n_variants):
        description += f" {br.prob_greater_than_z(threshold, outcome_metric)[variant]*100}% for variant {variant}"  # noqa: E501
        description += "." if variant == ed.n_variants - 1 else ",\n"
    print(description)

    # rope probability
    probs, rope_lower, rope_upper = br.rope(outcome_metric)

    description = "The probability that the average treatment effect is outside"
    " the region of practical equivalence"
    f" ({rope_lower},{rope_upper}) for {outcome_metric} is\n"
    for variant in range(1, ed.n_variants):
        description += f" {probs[variant]*100:.2f}% for variant {variant}"
        description += "." if variant == ed.n_variants - 1 else ",\n"
    print(description)


if __name__ == "__main__":
    pytest.main([__file__])
