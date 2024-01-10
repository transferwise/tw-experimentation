from tw_experimentation.variance_reduction.variance_reduction_method import (
    VarianceReductionMethod,
)
import pandas as pd


class DifferenceInMeans(VarianceReductionMethod):
    """Implement Difference-in-Means (DiM) estimator."""

    def fit(
        self, data: pd.DataFrame, treatment_column: str, target_column: str, **kwargs
    ):
        """Apply Difference In Means to data.

        Args:
            data (pd.DataFrame): experiment data containing pre-experiment data column
            treatment_column (str): name of column containing treatment flags
            target_column (str): name of column containing target metric

        Returns:
            DifferenceInMeans: self
        """

        # fit difference-in-means
        self.fit_baseline(
            data=data, treatment_column=treatment_column, target_column=target_column
        )

        # copy from baseline
        self.regression_results = self.baseline_results
        self.estimate = self.baseline_estimate
        self.p_value = self.baseline_p_value
        self.conf_int_95 = self.baseline_conf_int_95
        self.variance_reduction_rate = 0

        return self

    def calculate_variance_reduction(self):
        raise NotImplementedError

    def ci_width(self, alpha: float = 0.05):
        """Calculate the `1-alpha * 100%` confidence interval."""

        return (
            self.regression_results.conf_int(alpha=alpha, cols=None)[1][1]
            - self.regression_results.conf_int(alpha=alpha, cols=None)[0][1]
        )

    def calculate_ci_width_reduction(self, alpha=0.05):
        raise NotImplementedError
