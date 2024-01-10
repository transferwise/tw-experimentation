from abc import ABC, abstractmethod
import statsmodels.api as sm
import pandas as pd


class VarianceReductionMethod(ABC):
    """An abstract class for variance reduction methods."""

    def __init__(self):
        self.estimate = None
        self.variance_reduction_rate = None
        self.baseline_results = None

        self.estimate = None
        self.p_value = None
        self.conf_int_95 = None

    @abstractmethod
    def fit():
        pass

    @abstractmethod
    def calculate_variance_reduction():
        pass

    @abstractmethod
    def ci_width():
        pass

    @abstractmethod
    def calculate_ci_width_reduction(self, alpha=0.05):
        pass

    def fit_baseline(
        self, data: pd.DataFrame, treatment_column: str, target_column: str
    ):
        target = data[target_column]
        treatment = data[treatment_column]

        self.baseline_results = sm.OLS(
            target.to_numpy().astype(float),
            sm.add_constant(treatment.to_numpy().astype(float)),
        ).fit()
        self.baseline_estimate = self.baseline_results.params[1]
        self.baseline_p_value = self.baseline_results.pvalues[1]
        self.baseline_conf_int_95 = (
            self.baseline_results.conf_int(alpha=0.05, cols=None)[0][1],
            self.baseline_results.conf_int(alpha=0.05, cols=None)[1][1],
        )

    def calculate_bias(self, true, estimated):
        """Calculate bias as a fraction of true estimate."""

        return (estimated - true) / true
