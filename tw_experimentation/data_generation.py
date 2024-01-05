import numpy as np
import pandas as pd


from numpy import random
from dataclasses import dataclass
import datetime
from random import randint


@dataclass
class DGP:
    n: int = 1000

    def generate_data(self, seed: int = 0, **kwargs) -> pd.DataFrame:
        df = pd.DataFrame({"Y": [], "T": []})
        return df

    def generate_baseline(self) -> pd.DataFrame:
        return pd.DataFrame()

    def generate_treatment_effect(self) -> pd.DataFrame:
        return pd.DataFrame()

    def resample_variant_assignment(self) -> pd.DataFrame:
        return pd.DataFrame()

    def resample_treatment_effect(self) -> pd.DataFrame:
        return pd.DataFrame()


class SimpleConversionComparison(DGP):
    def generate_data(
        self,
        n=None,
        treatment_share: float = 0.5,
        baseline_conversion: float = 0.5,
        treatment_effect: float = 0.1,
        seed: int = 1,
    ) -> pd.DataFrame:
        treatment_expectation = baseline_conversion + treatment_effect
        if treatment_expectation > 1 or treatment_expectation < 0:
            raise Exception(
                "Conversion probability must be between 0 and 1 for treatment."
            )

        if n is not None:
            self.n = n

        np.random.seed(seed)
        variant = random.binomial(n=1, p=treatment_share, size=self.n)
        outcome = random.binomial(
            n=1, p=variant * treatment_effect + baseline_conversion
        )

        trigger_dates = [
            datetime.datetime(2022, 1, 1)
            + datetime.timedelta(
                days=randint(0, 100), hours=randint(0, 24), minutes=randint(0, 60)
            )
            for _ in range(self.n)
        ]

        df = pd.DataFrame({"T": variant, "Y": outcome, "date": trigger_dates})
        return df


class SimpleRevenueComparison(DGP):
    def generate_data(
        self,
        n=None,
        treatment_share: float = 0.5,
        baseline_mean: float = 50,
        sigma: float = 30,
        treatment_effect: float = 0.1,
        seed: int = 0,
    ) -> pd.DataFrame:
        if n is not None:
            self.n = n

        np.random.seed(seed)
        variant = random.binomial(n=1, p=treatment_share, size=self.n)
        outcome = random.lognormal(
            mean=variant * treatment_effect + baseline_mean, sigma=sigma
        )

        trigger_dates = [
            datetime.datetime(2022, 1, 1)
            + datetime.timedelta(
                days=randint(0, 100), hours=randint(0, 24), minutes=randint(0, 60)
            )
            for _ in range(self.n)
        ]

        df = pd.DataFrame({"T": variant, "Y": outcome, "date": trigger_dates})
        return df


class RevenueConversion(DGP):
    def generate_data(
        self,
        n=None,
        treatment_share: float = 0.5,
        baseline_conversion: float = 0.5,
        treatment_effect_conversion: float = 0.1,
        baseline_mean_revenue: float = 5,
        sigma_revenue: float = 2,
        treatment_effect_revenue=0.1,
        is_dynamic_assignment: bool = True,
        has_date_column: bool = True,
        seed: int = 1,
    ) -> pd.DataFrame:
        if n is not None:
            self.n = n

        treatment_expectation = baseline_conversion + treatment_effect_conversion
        if treatment_expectation > 1 or treatment_expectation < 0:
            raise Exception(
                "Conversion probability must be between 0 and 1 for treatment."
            )

        np.random.seed(seed)
        variant = random.binomial(n=1, p=treatment_share, size=self.n)
        conversion = random.binomial(
            n=1, p=variant * treatment_effect_conversion + baseline_conversion
        )
        revenue = random.lognormal(
            mean=variant * treatment_effect_revenue + baseline_mean_revenue,
            sigma=sigma_revenue,
        )

        pre_exp_revenue = random.lognormal(
            mean=baseline_mean_revenue, sigma=sigma_revenue, size=(self.n,)
        )

        num_actions = random.poisson(lam=8, size=self.n) * conversion

        if has_date_column and is_dynamic_assignment:
            trigger_dates = [
                datetime.datetime(2022, 1, 1)
                + datetime.timedelta(
                    days=randint(0, 20), hours=randint(0, 24), minutes=randint(0, 60)
                )
                for _ in range(self.n)
            ]
        elif has_date_column and not is_dynamic_assignment:
            trigger_dates = [datetime.datetime(2022, 1, 1)] * self.n

        df = pd.DataFrame(
            {
                "T": variant,
                "conversion": conversion,
                "revenue": revenue,
                "pre_exp_revenue": pre_exp_revenue,
                "num_actions": num_actions,
            }
        )
        if has_date_column:
            df["trigger_dates"] = trigger_dates
        return df

    def generate_data_abn_test(
        self,
        n_treatments=2,
        n=None,
        baseline_conversion: float = 0.5,
        treatment_effect_conversion: float = 0.01,
        baseline_mean_revenue: float = 1,
        sigma_revenue: float = 2,
        treatment_effect_revenue: float = 0.05,
        seed: int = 1,
    ) -> pd.DataFrame:
        # treatment effect is treatment_effect * variant

        if n is not None:
            self.n = n

        np.random.seed(seed)
        variant = random.randint(0, high=n_treatments + 1, size=self.n)
        conversion = random.binomial(
            n=1, p=variant * treatment_effect_conversion + baseline_conversion
        )

        revenue = conversion * random.lognormal(
            mean=variant * treatment_effect_revenue + baseline_mean_revenue,
            sigma=sigma_revenue,
        )

        pre_exp_revenue = random.lognormal(
            mean=baseline_mean_revenue,
            sigma=sigma_revenue,
            size=(self.n,),
        )

        num_actions = random.poisson(lam=8, size=self.n) * conversion

        trigger_dates = [
            datetime.datetime(2022, 1, 1)
            + datetime.timedelta(
                days=randint(0, 20), hours=randint(0, 24), minutes=randint(0, 60)
            )
            for _ in range(self.n)
        ]

        df = pd.DataFrame(
            {
                "T": variant,
                "conversion": conversion,
                "revenue": revenue,
                "pre_exp_revenue": pre_exp_revenue,
                "num_actions": num_actions,
                "trigger_dates": trigger_dates,
            }
        )
        return df


class SimpleClickThroughRate(DGP):
    """To test ratio metrics"""

    def generate_data(
        self,
        n_users: int = 1000,
        visits_lam: float = 10,
        visits_lam_treat_effect: float = 2,
        baseline_clickthrough=0.5,
        treatment_effect_clickthrough=0.1,
        treatment_share: float = 0.5,
        seed: int = 0,
    ) -> pd.DataFrame:
        random.seed(seed)

        variant = random.binomial(n=1, p=treatment_share, size=n_users)
        visits = random.poisson(lam=visits_lam + variant * visits_lam_treat_effect)

        clicks = random.binomial(
            n=visits, p=baseline_clickthrough + variant * treatment_effect_clickthrough
        )
        ctr = clicks / visits

        trigger_dates = [
            datetime.datetime(2022, 1, 1)
            + datetime.timedelta(
                days=randint(0, 100), hours=randint(0, 24), minutes=randint(0, 60)
            )
            for _ in range(self.n)
        ]

        df = pd.DataFrame(
            {
                "Y": ctr,
                "T": variant,
                "clicks": clicks,
                "visits": visits,
                "date": trigger_dates,
            }
        )
        return df


class MixtureModel(DGP):
    n_models_p_variant: int = 100

    def generate_data(
        self,
        n_models=100,
        n_per_model=200,
        seed: int = 0,
        ctrl_distribution="uniform",
        treat_distribution="beta",
        ctrl_dist_args=(),
        ctrl_dist_kwargs={},
        treat_dist_args=(1, 2),
        treat_dist_kwargs={},
        final_distribution="binomial",
        mix_param="p",
        final_dist_params_ctrl={"n": 1},
    ) -> pd.DataFrame:
        random.seed(seed)

        models_control = self.generate_models(
            *ctrl_dist_args,
            distribution=ctrl_distribution,
            n_models=n_models,
            **ctrl_dist_kwargs
        )
        # mu = .6
        # beta = 1
        # alpha = beta*mu/(1-mu)
        # args = (alpha, beta)
        models_treat = self.generate_models(
            *treat_dist_args,
            distribution=treat_distribution,
            n_models=n_models,
            **treat_dist_kwargs
        )

        samples = {}
        for v in [models_control, models_treat]:
            sampler = getattr(random, final_distribution)
            samples[str(v)] = sampler(
                **final_dist_params_ctrl,
                **{mix_param: v},
                size=(n_per_model, v.shape[0])
            )

        df_control = pd.DataFrame(
            {
                "Y": samples[str(models_control)].flatten(),
                "T": np.zeros(n_models * n_per_model),
            }
        )
        df_treat = pd.DataFrame(
            {
                "Y": samples[str(models_treat)].flatten(),
                "T": np.ones(n_models * n_per_model),
            }
        )
        df = pd.concat([df_control, df_treat])
        return df

    def generate_models(
        self, *args, n_models=100, distribution=None, **kwargs
    ) -> np.ndarray:
        if distribution == "beta":
            sampler = getattr(random, distribution)
            samples = sampler(*args, **kwargs, size=(n_models))
        else:
            sampler = getattr(random, distribution)
            samples = sampler(size=(n_models))

        return samples


class SimpleHeterogeneousEffect(DGP):
    def generate_data(
        self, seed: int = 1, treatment_share=0.3, n=None, true_te=False
    ) -> pd.DataFrame:
        if n is not None:
            self.n = n

        np.random.seed(seed)

        # feature
        age = np.round(np.random.uniform(18, 70, self.n), 2)

        # Treatment
        variant = np.random.binomial(1, treatment_share, self.n) == 1

        # Heterogeneous effects by age
        y0 = 10 + 0.1 * (30 < age) * (age < 50)
        y1 = 0.5 + 0.3 * (35 < age) * (age < 45)

        # Outcome
        revenue = np.round(np.random.normal(y0 + variant * y1, 0.15, self.n), 2)

        # timestamps
        trigger_dates = [
            datetime.datetime(2022, 1, 1)
            + datetime.timedelta(
                days=randint(0, 100), hours=randint(0, 24), minutes=randint(0, 60)
            )
            for _ in range(self.n)
        ]

        df = pd.DataFrame(
            {"Y": revenue, "T": variant, "feature": age, "date": trigger_dates}
        )

        # Add truth
        if true_te:
            df["y0"] = y0
            df["y1"] = y1

        return df


class GenerativeFromRealData(DGP):
    """Generative DGP from real data using residuals
    Requires fitted autocausality model / predictions as input
    """

    pass


class SimpleExperimentOneConfounder(DGP):
    """
    Synthetic experiment with one observed confounder ('web' or 'mobile' user)
    """

    pass
