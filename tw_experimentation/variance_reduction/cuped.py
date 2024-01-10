import numpy as np
import statsmodels.api as sm
import pandas as pd
from tw_experimentation.variance_reduction.variance_reduction_method import (
    VarianceReductionMethod,
)
from typing import List


class CUPED(VarianceReductionMethod):
    """Implement Controlled-experiment Using Pre-Existing Data (CUPED)."""

    def fit(
        self,
        data: pd.DataFrame,
        treatment_column: str,
        target_column: str,
        covariate_column: str,
        **kwargs
    ):
        """Apply CUPED to data.

        Args:
            data (pd.DataFrame): experiment data containing pre-experiment data column
            treatment_column (str): name of column containing treatment flags
            target_column (str): name of column containing target metric
            covariate_column (str): name of column containing the covariate (pre-experiment data)

        Returns:
            CUPED: self
        """

        target = data[target_column]
        covariate = data[covariate_column]

        assert set(data[treatment_column].unique()) == {0, 1}
        treatment = data[treatment_column]

        # treatment = pd.get_dummies(data[treatment_column], drop_first=True) # TODO: investigate behaviour when treatment is already binary, also investigate what happens when treatment has more than 2 distinct values

        # compute theta by regressing the target on the covariate
        t = (
            sm.OLS(
                target.to_numpy().astype(float),
                sm.add_constant(covariate.to_numpy().astype(float)),
            )
            .fit()
            .params
        )
        print(t)
        theta = (
            sm.OLS(
                target.to_numpy().astype(float),
                sm.add_constant(covariate.to_numpy().astype(float)),
            )
            .fit()
            .params[1]
        )

        # calculate the CUPED adjusted target metric
        cuped_adjusted_target = target - theta * (covariate - np.mean(covariate))

        # fit CUPED
        self.regression_results = sm.OLS(
            cuped_adjusted_target.astype(float),
            sm.add_constant(treatment.astype(float)),
        ).fit()

        # statistics
        self.estimate = self.regression_results.params[1]
        self.p_value = self.regression_results.pvalues[1]
        self.conf_int_95 = (
            self.regression_results.conf_int(alpha=0.05, cols=None)[0][1],
            self.regression_results.conf_int(alpha=0.05, cols=None)[1][1],
        )
        self.variance_reduction_rate = self.calculate_variance_reduction(
            target, cuped_adjusted_target
        )

        # fit difference-in-means
        self.fit_baseline(
            data=data, treatment_column=treatment_column, target_column=target_column
        )
        return self

    @staticmethod
    def ci_width(ols_results, alpha=0.05):  # TODO: typing
        """Calculate the `1-alpha * 100%` confidence interval."""

        return (
            ols_results.conf_int(alpha=alpha, cols=None)[1][1]
            - ols_results.conf_int(alpha=alpha, cols=None)[0][1]
        )

    @staticmethod
    def calculate_variance_reduction(
        target: pd.Series, target_cuped: pd.Series
    ) -> float:  # TODO: verify the input data types
        """Calculate variance reduction as a fraction of original variance."""

        frac = np.var(target_cuped) / np.var(target) if np.var(target) != 0 else 1

        return 1 - frac

    def calculate_ci_width_reduction(self, alpha: float = 0.05) -> float:
        """Calculate confidence interval width reduction."""

        frac = self.ci_width(self.regression_results, alpha) / self.ci_width(
            self.baseline_results, alpha
        )

        return 1 - frac


def multiple_CUPEDs(
    data: pd.DataFrame,
    treatment_column: str,
    target_columns: List[str],
    covariate_columns: List[str],
    ci_alpha: float = 0.05,
):  # TODO: typing
    """Applies CUPED to multiple target--covariate pairs."""

    n_target_cols = len(target_columns)
    n_covariate_cols = len(covariate_columns)

    cuped_matrix = np.ndarray((n_target_cols, n_covariate_cols), dtype=object)
    estimates_matrix = np.zeros((n_target_cols, n_covariate_cols))
    p_values_matrix = np.ndarray((n_target_cols, n_covariate_cols), dtype=object)
    variance_reduction_rates_matrix = np.zeros((n_target_cols, n_covariate_cols))
    ci_width_reduction_rates_matrix = np.zeros((n_target_cols, n_covariate_cols))

    # apply CUPED to each target--covariate pair and store evaluation metrics
    for i, target_col in enumerate(target_columns):
        for j, covariate_col in enumerate(covariate_columns):
            cuped_matrix[i, j] = CUPED().fit(
                data, treatment_column, target_col, covariate_col
            )
            estimates_matrix[i, j] = cuped_matrix[i, j].estimate
            p_values_matrix[i, j] = (
                cuped_matrix[i, j].baseline_p_value,
                cuped_matrix[i, j].p_value,
            )
            variance_reduction_rates_matrix[i, j] = cuped_matrix[
                i, j
            ].variance_reduction_rate
            ci_width_reduction_rates_matrix[i, j] = cuped_matrix[
                i, j
            ].calculate_ci_width_reduction(ci_alpha)

    return (
        cuped_matrix,
        estimates_matrix,
        variance_reduction_rates_matrix,
        ci_width_reduction_rates_matrix,
        p_values_matrix,
    )  # TODO: change output data to dictionary


# =================================================================================================
# =================================================================================================
# =================================================================================================


class MultivariateCUPED(
    VarianceReductionMethod
):  # NOTE: THIS IS COPIED FROM AMBROSIA, TODO: CHECK REFERENCING
    """Implements the multivariate version of CUPED."""

    def fit(
        self,
        data: pd.DataFrame,
        treatment_column: str,
        target_column: str,
        covariate_columns: List[str],
        **kwargs
    ):
        """Fit the multivariate CUPED model."""

        target = data[target_column]
        covariates = data[covariate_columns]

        assert set(data[treatment_column].unique()) == {0, 1}
        treatment = data[treatment_column]

        # treatment = pd.get_dummies(data[treatment_column], drop_first=True) # TODO: investigate behaviour when treatment is already binary, also investigate what happens when treatment has more than 2 distinct values

        # compute theta by regressing the target on the covariate
        covariance = data[[target_column] + covariate_columns].cov()
        matrix = covariance.loc[covariate_columns, covariate_columns]
        num_features = len(covariate_columns)
        covariance_target = covariance.loc[
            covariate_columns, target_column
        ].values.reshape(num_features, -1)

        theta = np.linalg.pinv(matrix) @ covariance_target  # NOTE: using pseudoinverse
        means = (data[covariate_columns].values @ theta).reshape(-1).mean()

        # calculate the CUPED adjusted target metric
        cuped_adjusted_target = target - (covariates.values @ theta).reshape(-1) + means

        # fit CUPED
        self.regression_results = sm.OLS(
            cuped_adjusted_target,
            sm.add_constant(treatment.to_numpy().astype(float)),
        ).fit()

        # statistics
        self.estimate = self.regression_results.params[1]
        self.p_value = self.regression_results.pvalues[1]
        self.conf_int_95 = (
            self.regression_results.conf_int(alpha=0.05, cols=None)[0][1],
            self.regression_results.conf_int(alpha=0.05, cols=None)[1][1],
        )
        self.variance_reduction_rate = self.calculate_variance_reduction(
            target, cuped_adjusted_target
        )

        # fit difference-in-means
        self.fit_baseline(
            data=data, treatment_column=treatment_column, target_column=target_column
        )
        return self

    @staticmethod
    def ci_width(ols_results, alpha=0.05):  # TODO: typing
        """Calculate the `1-alpha * 100%` confidence interval."""

        return (
            ols_results.conf_int(alpha=alpha, cols=None)[1][1]
            - ols_results.conf_int(alpha=alpha, cols=None)[0][1]
        )

    @staticmethod
    def calculate_variance_reduction(
        target: pd.Series, target_cuped: pd.Series
    ) -> float:  # TODO: verify the input data types
        """Calculate variance reduction as a fraction of original variance."""

        frac = np.var(target_cuped) / np.var(target) if np.var(target) != 0 else 1

        return 1 - frac

    def calculate_ci_width_reduction(self, alpha: float = 0.05) -> float:
        """Calculate confidence interval width reduction."""

        frac = self.ci_width(self.regression_results, alpha) / self.ci_width(
            self.baseline_results, alpha
        )

        return 1 - frac
