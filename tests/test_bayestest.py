import pytest
import warnings

from tw_experimentation.utils import ExperimentDataset

from tw_experimentation.bayes.bayes_test import BayesTest
from tw_experimentation.data_generation import RevenueConversion
from tw_experimentation.bayes.numpyro_monkeypatch import ZeroInflatedLogNormal

from numpyro.distributions import LogNormal, Gamma, Uniform


class TestBayesTesterEndToEnd(object):
    def test_with_new_model(self):
        rc = RevenueConversion()
        df = rc.generate_data(
            sigma_revenue=1,
            treatment_effect_revenue=0.1,
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

        bt = BayesTest(ed=ed)

        # bt.set_model(
        #     "revenue",
        #     ZeroInflatedLogNormal,
        #     ["loc", "scale", "gate"],
        #     [LogNormal, Gamma, Uniform],
        #     [
        #         {"args": {}, "kwargs": {"loc": 0, "scale": 1}},
        #         {"args": (1,), "kwargs": {"rate": 0.5}},
        #         {"args": (), "kwargs": {"low": 0, "high": 1}},
        #     ],
        # )

        br = bt.compute_posterior(verbose=1)

        br.fig_posterior_difference_by_target("revenue")

        br.prob_greater_than_zero("revenue")

        br.prob_greater_than_z(5, "revenue")

        br.rope("revenue", 10, -10)

        br.bayes_factor("revenue", 1)

        br.bayes_factor_decision("revenue", 1)


if __name__ == "__main__":
    pytest.main([__file__])
