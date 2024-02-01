from typing import List

import pandas as pd
import statsmodels.api as sm

from tw_experimentation.variance_reduction.variance_reduction_method import (
    VarianceReductionMethod,
)


class MultivariateRegression(VarianceReductionMethod):
    """Multivariate regression variance reduction method."""

    def fit(
        self,
        data: pd.DataFrame,
        treatment_column: str,
        target_column: str,
        covariate_columns: List[str],
        **kwargs,
    ):
        """Apply Multivariate Regression to data.

        Args:
            data (pd.DataFrame): experiment data
            treatment_column (str): name of column containing treatment flags
            target_column (str): name of column containing target metric
            covariate_columns (List[str]): list of names of covariate columns

        Returns:
            MultivariateRegression: self
        """

        self.treatment_column = treatment_column

        covariates = data[covariate_columns]
        target = data[target_column]
        treatment = data[treatment_column]

        # merge together treatment and covariates
        X_data = covariates.assign(T=treatment)

        # rename column T
        X_data = X_data.rename(columns={"T": treatment_column})

        # fit multivariate regression
        self.regression_results = sm.OLS(target, sm.add_constant(X_data)).fit()

        # fit baseline
        self.fit_baseline(
            data=data, treatment_column=treatment_column, target_column=target_column
        )

        # statistics
        self.estimate = self.regression_results.params[treatment_column]
        self.p_value = self.regression_results.pvalues[treatment_column]
        self.conf_int_95 = (
            self.regression_results.conf_int(alpha=0.05, cols=None)[0][
                treatment_column
            ],
            self.regression_results.conf_int(alpha=0.05, cols=None)[1][
                treatment_column
            ],
        )
        self.variance_reduction_rate = self.calculate_variance_reduction()

        return self

    def calculate_variance_reduction(self):
        """Calculate the variance reduction rate."""

        frac = (
            self.regression_results.bse[self.treatment_column]
            / self.baseline_results.bse[1]
        )

        return 1 - frac**2

    def ci_width(self, alpha: float = 0.05):
        """Calculate the `1-alpha * 100%` confidence interval."""

        return (
            self.regression_results.conf_int(alpha=alpha, cols=None)[1][
                self.treatment_column
            ]
            - self.regression_results.conf_int(alpha=alpha, cols=None)[0][
                self.treatment_column
            ]
        )

    def calculate_ci_width_reduction(self, alpha=0.05):
        """Calculate the width reduction rate in 1-`alpha` * 100% confidence
        interval."""

        frac = self.ci_width(self.regression_results, alpha) / self.ci_width(
            self.baseline_results, alpha
        )

        return 1 - frac


# =================================================================================================
# =================================================================================================
# =================================================================================================


class MultivariateRegressionAdjusted(VarianceReductionMethod):
    def fit(
        self,
        data: pd.DataFrame,
        treatment_column: str,
        target_column: str,
        covariate_columns: List[str],
    ):
        """Apply Multivariate Regression to data.

        Args:
            data (pd.DataFrame): experiment data
            treatment_column (str): name of column containing treatment flags
            target_column (str): name of column containing target metric
            covariate_columns (List[str]): list of names of covariate columns

        Returns:
            MultivariateRegression: self
        """

        self.treatment_column = treatment_column

        covariates = data[covariate_columns]
        target = data[target_column]
        treatment = data[treatment_column]

        # prepare adjustment
        X_data = covariates.copy()
        # for col in covariate_columns:
        #     X_data[col+'_tilde'] = treatment * (X_data[col] - X_data[col].mean())

        X_data = pd.concat(
            [X_data]
            + [
                (treatment * (X_data[col] - X_data[col].mean())).rename(col + "_tilde")
                for col in covariate_columns
            ],
            axis=1,
        )

        # merge together treatment and covariates
        X_data = X_data.assign(T=treatment)

        # rename column T
        X_data = X_data.rename(columns={"T": treatment_column})

        # fit multivariate regression
        self.regression_results = sm.OLS(target, sm.add_constant(X_data)).fit()

        # fit baseline
        self.fit_baseline(
            data=data, treatment_column=treatment_column, target_column=target_column
        )

        # statistics
        self.estimate = self.regression_results.params[treatment_column]
        self.p_value = self.regression_results.pvalues[treatment_column]
        self.conf_int_95 = (
            self.regression_results.conf_int(alpha=0.05, cols=None)[0][
                treatment_column
            ],
            self.regression_results.conf_int(alpha=0.05, cols=None)[1][
                treatment_column
            ],
        )
        self.variance_reduction_rate = self.calculate_variance_reduction()

        return self

    def calculate_variance_reduction(self):
        """Calculate the variance reduction rate."""

        frac = (
            self.regression_results.bse[self.treatment_column]
            / self.baseline_results.bse[1]
        )

        return 1 - frac**2

    def ci_width(self, alpha: float = 0.05):
        """Calculate the `1-alpha * 100%` confidence interval."""

        return (
            self.regression_results.conf_int(alpha=alpha, cols=None)[1][
                self.treatment_column
            ]
            - self.regression_results.conf_int(alpha=alpha, cols=None)[0][
                self.treatment_column
            ]
        )

    def calculate_ci_width_reduction(self, alpha=0.05):
        """Calculate the width reduction rate in 1-`alpha` * 100% confidence
        interval."""

        frac = self.ci_width(self.regression_results, alpha) / self.ci_width(
            self.baseline_results, alpha
        )

        return 1 - frac
