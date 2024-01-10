from tw_experimentation.variance_reduction.variance_reduction_method import (
    VarianceReductionMethod,
)
from tw_experimentation.variance_reduction.cuped import (
    CUPED,
)
from typing import List
from typing_extensions import Self
from sklearn.ensemble import HistGradientBoostingRegressor
import pandas as pd


class CUPAC(CUPED):
    """Implement Control Using Predictions As Covariates (CUPAC)."""

    def fit(
        self,
        data: pd.DataFrame,
        pre_experiment_data: pd.DataFrame,
        treatment_column: str,
        target_column: str,
        covariate_columns: List[str],
        model=HistGradientBoostingRegressor,  # TODO: typing
        model_init_config: dict = {},
        model_fit_config: dict = {},
        **kwargs
    ) -> Self:
        """Applies CUPAC to data.

        Args:
            data (pd.DataFrame): experiment data
            pre_experiment_data (pd.DataFrame): pre-experiment data
            treatment_column (str): name of the column containing the treatment flag
            target_column (str): name of the column containing the target metric
            covariate_columns (List[str]): list of names of covariate columns
            model (str, optional): regression model. Defaults to 'HistGBoost'.

        Raises:
            NotImplementedError: when an unsupported regression modelis used

        Returns:
            CUPAC: self
        """

        X_train = pre_experiment_data[covariate_columns]
        y_train = pre_experiment_data[target_column]
        X_pred = data[covariate_columns]

        # fit the regressor
        regressor = model(**model_init_config)
        regressor.fit(X_train, y_train, **model_fit_config)
        y_pred = regressor.predict(X_pred)

        # construct a dataframe with target, treatment and cupac predictions
        cupac_df = data[[treatment_column, target_column]]
        cupac_df = cupac_df.assign(cupac_covariate=y_pred)

        # fit a CUPED on the constructed data
        super().fit(
            data=cupac_df,
            treatment_column=treatment_column,
            target_column=target_column,
            covariate_column="cupac_covariate",
        )

        return self
