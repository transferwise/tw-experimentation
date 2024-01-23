import numpyro.distributions as dist
import pytest
from jax import random
from numpyro.infer import MCMC, NUTS

from tw_experimentation.bayes.bayes_model import BayesModel
from tw_experimentation.data_generation import RevenueConversion
from tw_experimentation.utils import ExperimentDataset


class TestNumpyroSampler(object):
    @pytest.fixture
    def data_input(self):
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

    @pytest.mark.skip(reason="need to adjust test to refactored code")
    def test_binary_adjust_prior_param(self, data_input):
        ed = data_input
        bm = BayesModel(
            metric_type="binary", observations=ed.data.groupby("T")["conversion"]
        )
        nuts = NUTS(bm.build_model, adapt_step_size=True)
        mcmc = MCMC(nuts, num_samples=1000, num_warmup=500)
        rng_key = random.PRNGKey(0)

        bm.set_prior_model_param("probs", {"args": (5, 4), "kwargs": {}})

        mcmc.run(
            rng_key,
        )
        mcmc.print_summary(exclude_deterministic=False)

    @pytest.mark.skip(reason="need to adjust test to refactored code")
    def test_binary_adjust_prior_model(self, data_input):
        ed = data_input
        bm = BayesModel(
            metric_type="binary", observations=ed.data.groupby("T")["conversion"]
        )
        nuts = NUTS(bm.build_model, adapt_step_size=True)
        mcmc = MCMC(nuts, num_samples=1000, num_warmup=500)
        rng_key = random.PRNGKey(0)

        bm.set_prior_model(
            "probs", dist.Uniform, {"args": (), "kwargs": {"low": 0, "high": 1}}
        )

        mcmc.run(
            rng_key,
        )
        mcmc.print_summary(exclude_deterministic=False)

    @pytest.mark.skip(reason="need to adjust test to refactored code")
    def test_continuous_full_model(self, data_input):
        ed = data_input
        bm = BayesModel(
            metric_type="continuous", observations=ed.data.groupby("T")["revenue"]
        )
        nuts = NUTS(bm.build_model, adapt_step_size=True)
        mcmc = MCMC(nuts, num_samples=1000, num_warmup=500)
        rng_key = random.PRNGKey(0)

        # bm.set_model(
        #     dist.Normal,
        #     ["loc", "scale"],
        #     [dist.LogNormal, dist.Gamma],
        #     [
        #         {"args": {}, "kwargs": {"loc": 0, "scale": 1}},
        #         {"args": (1,), "kwargs": {"rate": 0.5}},
        #     ],
        # )

        mcmc.run(
            rng_key,
        )
        mcmc.print_summary(exclude_deterministic=False)


if __name__ == "__main__":
    pytest.main([__file__])
