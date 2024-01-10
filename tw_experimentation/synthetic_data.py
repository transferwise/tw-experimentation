import numpy as np
import pandas as pd


from numpy import random
import datetime
from random import randint


def revenue_conversion_data(
    n=1000,
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
    """
    Generate synthetic revenue and conversion metrics data for experimentation.

    Args:
        n (int): Number of observations to generate (default: 1000).
        treatment_share (float): Share of observations assigned to treatment (default: 0.5).
        baseline_conversion (float): Baseline conversion rate (default: 0.5).
        treatment_effect_conversion (float): Treatment effect on conversion rate (default: 0.1).
        baseline_mean_revenue (float): Baseline mean revenue (default: 5).
        sigma_revenue (float): Standard deviation of revenue (default: 2).
        treatment_effect_revenue (float): Treatment effect on revenue (default: 0.1).
        is_dynamic_assignment (bool): Whether to assign treatment dynamically based on trigger dates (default: True).
        has_date_column (bool): Whether to include a date column in the output dataframe (default: True).
        seed (int): Random seed for reproducibility (default: 1).

    Returns:
        pd.DataFrame: Generated synthetic data with columns: 'T', 'conversion', 'revenue', 'pre_exp_revenue', 'num_actions'.
                      If 'has_date_column' is True, the dataframe will also include a 'trigger_dates' column.
    """

    treatment_expectation = baseline_conversion + treatment_effect_conversion
    if treatment_expectation > 1 or treatment_expectation < 0:
        raise Exception("Conversion probability must be between 0 and 1 for treatment.")

    np.random.seed(seed)
    variant = random.binomial(n=1, p=treatment_share, size=n)
    conversion = random.binomial(
        n=1, p=variant * treatment_effect_conversion + baseline_conversion
    )
    revenue = random.lognormal(
        mean=variant * treatment_effect_revenue + baseline_mean_revenue,
        sigma=sigma_revenue,
    )

    pre_exp_revenue = random.lognormal(
        mean=baseline_mean_revenue, sigma=sigma_revenue, size=(n,)
    )

    num_actions = random.poisson(lam=8, size=n) * conversion

    if has_date_column and is_dynamic_assignment:
        trigger_dates = [
            datetime.datetime(2022, 1, 1)
            + datetime.timedelta(
                days=randint(0, 20), hours=randint(0, 24), minutes=randint(0, 60)
            )
            for _ in range(n)
        ]
    elif has_date_column and not is_dynamic_assignment:
        trigger_dates = [datetime.datetime(2022, 1, 1)] * n

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


def revenue_conversion_abn_test(
    n_treatments=2,
    n=1000,
    baseline_conversion: float = 0.5,
    treatment_effect_conversion: float = 0.01,
    baseline_mean_revenue: float = 1,
    sigma_revenue: float = 2,
    treatment_effect_revenue: float = 0.05,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Generate synthetic data for revenue and conversion metrics in an ABN test.

    Args:
        n_treatments (int): Number of treatments (default: 2).
        n (int): Number of samples (default: None).
        baseline_conversion (float): Baseline conversion rate (default: 0.5).
        treatment_effect_conversion (float): Treatment effect on conversion rate (default: 0.01).
        baseline_mean_revenue (float): Baseline mean revenue (default: 1).
        sigma_revenue (float): Standard deviation of revenue (default: 2).
        treatment_effect_revenue (float): Treatment effect on revenue (default: 0.05).
        seed (int): Random seed (default: 1).

    Returns:
        pd.DataFrame: Synthetic data with columns: 'T', 'conversion', 'revenue', 'pre_exp_revenue', 'num_actions', 'trigger_dates'.
    """

    np.random.seed(seed)
    variant = random.randint(0, high=n_treatments + 1, size=n)
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
        size=(n,),
    )

    num_actions = random.poisson(lam=8, size=n) * conversion

    trigger_dates = [
        datetime.datetime(2022, 1, 1)
        + datetime.timedelta(
            days=randint(0, 20), hours=randint(0, 24), minutes=randint(0, 60)
        )
        for _ in range(n)
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


def click_through_data(
    n_users: int = 1000,
    visits_lam: float = 10,
    visits_lam_treat_effect: float = 2,
    baseline_clickthrough=0.5,
    treatment_effect_clickthrough=0.1,
    treatment_share: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Generate synthetic click-through data for an experiment.

    Args:
        n_users (int): Number of users.
        visits_lam (float): Average number of visits per user.
        visits_lam_treat_effect (float): Treatment effect on average visits per user.
        baseline_clickthrough (float): Baseline click-through rate.
        treatment_effect_clickthrough (float): Treatment effect on click-through rate.
        treatment_share (float): Share of users assigned to treatment.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing synthetic click-through data.

    """
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
        for _ in range(n_users)
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
