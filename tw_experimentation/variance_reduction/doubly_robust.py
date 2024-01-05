from econml.dr import LinearDRLearner
from tw_experimentation.variance_reduction.variance_reduction_method import (
    VarianceReductionMethod,
)
import pandas as pd
from typing import List
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(
    action="ignore", category=FutureWarning
)  # NOTE: should actually be investigated


class DoublyRobustEstimator(VarianceReductionMethod):
    """Implements a Doubly Robust Estimator for Causal Inference."""

    def fit(
        self,
        data: pd.DataFrame,
        treatment_column: str,
        target_column: str,
        covariate_columns: List[str],
        model_propensity="auto",
        model_propensity_init_config={},
        model_regression=HistGradientBoostingRegressor,
        model_regression_init_config={},
        **kwargs
    ):
        """Fit the Doubly Robust Estimator to the data.s

        Args:
            data (pd.DataFrame): experiment data
            treatment_column (str): name of the column containing the treatment flags
            target_column (str): name of the column containing the target metric
            covariate_columns (List[str]): list of names of covariate columns

        Returns:
            DoublyRobustEstimator: self
        """

        target = data[target_column]
        treatment = data[treatment_column]
        covariates = data[covariate_columns]

        # initialize models
        if model_propensity != "auto":
            initialized_model_propensity = model_propensity(
                **model_propensity_init_config
            )
        else:
            initialized_model_propensity = model_propensity
        initialized_model_regression = model_regression(**model_regression_init_config)

        # fit Doubly Robust Learner
        self.estimator = LinearDRLearner(
            model_propensity=initialized_model_propensity,
            model_regression=initialized_model_regression,
        )
        self.estimator.fit(Y=target, T=treatment, X=covariates, W=None)

        # fit a difference-in-means estimator
        self.fit_baseline(
            data=data, treatment_column=treatment_column, target_column=target_column
        )

        # generate statistics
        self.estimate = self.estimator.ate(
            X=covariates
        )  # TODO: should I use .effect(X=covariates) instead?
        self.conf_int_95 = self.conf_int(
            data=data, covariate_columns=covariate_columns, alpha=0.05
        )

        return self

    def calculate_variance_reduction():
        raise NotImplementedError("No explicit form, estimate through bootstrapping")

    def conf_int(self, data, covariate_columns, alpha: float = 0.05):
        """Calculate the 1-`alpha` * 100% confidence interval for the ATE estimate."""

        return self.estimator.ate_interval(
            X=data[covariate_columns].to_numpy(), alpha=alpha
        )

    def ci_width(self, data, covariate_columns, alpha: float = 0.05):
        """Calculate the width of the 1-`alpha` * 100% confidence interval."""

        ci = self.conf_int(data, covariate_columns, alpha=alpha)
        return ci[1] - ci[0]

    def calculate_ci_width_reduction(
        self, data, covariate_columns, alpha: float = 0.05
    ):
        """Calculate the width reduction rate in 1-`alpha` * 100% confidence interval."""

        baseline_conf_int = self.baseline_results.conf_int(alpha=alpha, cols=None)
        baseline_ci_width = baseline_conf_int[1][1] - baseline_conf_int[0][1]
        dr_ci_width = self.ci_width(data, covariate_columns, alpha=alpha)

        return 1 - (dr_ci_width / baseline_ci_width)
