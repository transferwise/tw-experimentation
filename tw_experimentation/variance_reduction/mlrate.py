from tw_experimentation.variance_reduction.variance_reduction_method import (
    VarianceReductionMethod,
)
from tw_experimentation.variance_reduction.utils import (
    split_dataframe,
)
from typing import List
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from flaml import AutoML
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold


class MLRATE(VarianceReductionMethod):
    """Implements Machine Learning Regression-Adjusted Treatment Effect Estimator (MLRATE)."""

    def fit(
        self,
        data: pd.DataFrame,
        K_splits: int,
        treatment_column: str,
        target_column: str,
        covariate_columns: List[str],
        model=AutoML,
        model_init_config: dict = {},
        model_fit_config: dict = {},
        **kwargs
    ):
        """_summary_

        Args:
            data (pd.DataFrame): experiment data
            K_splits (int): number of cross-fitting folds/splits
            treatment_column (str): name of column containing the treatment flag
            target_column (str): name of column containing the target metric
            covariate_columns (List[str]): list of names of covariate columns
            model (_type_, optional): regression model. Defaults to AutoML.
            model_init_config (dict, optional): configuration parameters passed at the initialization of the model. Defaults to {}.
            model_fit_config (dict, optional): configuration parameters passed at the fitting of the model. Defaults to {}.

        Returns:
            MLRATE: self
        """

        # reset indexing
        data = data.reset_index(drop=True)  # TODO: check if overwrites data

        assert set(data[treatment_column].unique()) == {0, 1}

        # split data into `K_splits`-folds for cross-fitting
        splits, index_to_split_map = split_dataframe(df=data, K=K_splits)
        # splits, index_to_split_map = split_dataframe(df=data.loc[data[treatment_column=0]], K=K_splits)
        N = len(data)

        target = data[target_column]
        treatment = data[treatment_column]
        covariates = data[covariate_columns]

        # initialization
        g_ks = [model(**model_fit_config) for _ in range(K_splits)]
        g_k_scores = [None for _ in range(K_splits)]

        # cross-fitting the regressors
        # print('fitting gs')
        for k, split_indices in enumerate(splits):
            all_indices = np.array([i for i in range(N)])
            indices_complement = np.setdiff1d(all_indices, split_indices)

            X_train = covariates.iloc[indices_complement]
            y_train = target.iloc[indices_complement]

            # g_k = regressor.fit(X_train, y_train, **model_fit_config)
            # g_ks.append(g_k)

            g_ks[k].fit(X_train, y_train, **model_fit_config)
            g_k_scores[k] = g_ks[k].score(X_train, y_train)

        g_pred = np.zeros(N)  # test set predictions

        # predicting the target
        # print('predicting')
        for split_indices, g_k in zip(splits, g_ks):
            g_k_pred = g_k.predict(covariates.iloc[split_indices])
            g_pred[split_indices] = g_k_pred

        self._diag = {
            "g_ks": g_ks,
            "g_pred": g_pred,
            "g_k_scores": g_k_scores,
            "splits": splits,
            "index_to_split_map": index_to_split_map,
            "treatment": treatment,
            "target": target,
            "covariates": covariates,
        }

        g_bar = g_pred.mean()  # mean of test set predictionss

        # assembling dataframe for OLS
        # print('assembling df')
        ml_rate_columns = {
            "target": target,
            treatment_column: treatment,
            "g_pred": g_pred,
            "g_pred_difference": np.multiply(treatment, g_pred - g_bar),
        }

        ml_rate_df = pd.DataFrame(ml_rate_columns)

        # OLS estimator
        # print('regressing')
        self.regression_results = sm.OLS(
            ml_rate_df["target"],
            sm.add_constant(
                ml_rate_df[[treatment_column, "g_pred", "g_pred_difference"]]
            ),
        ).fit()

        # self.regression_results = sm.OLS(ml_rate_df['target'],
        #                                  sm.add_constant(ml_rate_df[[treatment_column, 'g_pred']]))\
        #                             .fit()

        # fit difference-in-means estimator
        self.fit_baseline(
            data=data, treatment_column=treatment_column, target_column=target_column
        )

        # generate statistics
        # print('calculating statistics')
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
        self.robust_variance_estimate = self.robust_variance_estimator(
            Y=target.to_numpy(), T=treatment.to_numpy(), g_pred=g_pred
        )
        return self

    # TODO: Fix! This does not seem to be working properly
    @staticmethod
    def robust_variance_estimator(Y: np.array, T: np.array, g_pred: np.array):
        """Compute the robust variance estimator

        Args:
            Y (np.array): target metric vector
            T (np.array): treatment vector
            g_pred (np.array): vector of predictions of the target metric

        Returns:
            float: variance estimate
        """

        N = len(Y)

        p_hat = T.sum() / N

        Y_0 = Y[T == 0]
        Y_1 = Y[T == 1]

        var_term_0 = np.var(Y_0) / (1 - p_hat)
        var_term_1 = np.var(Y_1) / (p_hat)
        var_term_g = np.var(g_pred) / ((1 - p_hat) * p_hat)

        Z = np.column_stack([np.ones(N), T, g_pred, T * g_pred])

        beta_hat = np.linalg.inv(Z.T @ Z / N) @ (Y.T @ Z / N)

        beta_2_term = beta_hat[2]
        beta_3_term = beta_hat[3]

        sigma_hat_sqrd = (
            var_term_0
            + var_term_1
            - var_term_g
            * (beta_2_term * p_hat + (beta_2_term + beta_3_term) * (1 - p_hat))
        )

        return sigma_hat_sqrd

    def calculate_variance_reduction(self):
        """Calculate variance reduction for method

        Returns:
            float: variance reduction rate
        """

        frac = (
            self.regression_results.bse[1] / self.baseline_results.bse[1]
        )  # NOTE: regular, not robust errors

        return 1 - frac**2

    @staticmethod
    def ci_width(ols_results, alpha=0.05):  # TODO: typing
        """Calculate the `1-alpha * 100%` confidence interval."""

        return (
            ols_results.conf_int(alpha=alpha, cols=None)[1][1]
            - ols_results.conf_int(alpha=alpha, cols=None)[0][1]
        )

    def calculate_ci_width_reduction(self, alpha: float = 0.05) -> float:
        """Calculate confidence interval width reduction."""

        frac = self.ci_width(self.regression_results, alpha) / self.ci_width(
            self.baseline_results, alpha
        )

        return 1 - frac


# ==============================================================================
# ==============================================================================
# ==============================================================================

# https://github.com/muratunalphd/Blog-Posts/blob/main/variance-reduction-methods/MLRATE.ipynb


class AltMLRATE(VarianceReductionMethod):
    def fit(
        self,
        data: pd.DataFrame,
        K_splits: int,
        treatment_column: str,
        target_column: str,
        covariate_columns: List[str],
        **kwargs
    ):
        dfml = data.reset_index(drop=True)  # TODO: check if overwrites data

        XGB_reg = XGBRegressor(
            learning_rate=0.1, max_depth=6, n_estimators=500, reg_lambda=1
        )

        # X_mat = dfml.columns.tolist()[0:p]
        X_mat = covariate_columns

        kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)

        ix = []
        Yhat = []
        for train_index, test_index in kfold.split(dfml, dfml[treatment_column]):
            df_train = dfml.iloc[train_index].reset_index()
            df_test = dfml.iloc[test_index].reset_index()

            X_train = df_train[X_mat].copy()
            y_train = df_train[target_column].copy()
            X_test = df_test[X_mat].copy()
            XGB_reg.fit(X_train, y_train)

            Y_hat = XGB_reg.predict(X_test)

            ix.extend(list(test_index))
            Yhat.extend(list(Y_hat))

        df_ml = (
            pd.DataFrame({"ix": ix, "Yhat": Yhat})
            .sort_values(by="ix")
            .reset_index(drop=True)
        )
        df_ml[[target_column, treatment_column]] = dfml[
            [target_column, treatment_column]
        ]
        df_ml["Ytilde"] = df_ml["Yhat"] - np.mean(df_ml["Yhat"])
        df_ml["Yres"] = df_ml[target_column] - df_ml["Yhat"]
        df_ml = df_ml.drop("ix", axis=1)

        # mlrate =  smf.ols('Y ~ T + Yhat + T:Ytilde',
        #              data = df_ml).fit(cov_type='HC1',use_t=True)

        mlrate = smf.ols(
            target_column
            + " ~ "
            + treatment_column
            + " + Yhat + "
            + treatment_column
            + ":Ytilde",
            data=df_ml,
        ).fit(cov_type="HC1", use_t=True)

        self.regression_results = mlrate

        # fit difference-in-means estimator
        self.fit_baseline(
            data=data, treatment_column=treatment_column, target_column=target_column
        )

        # generate statistics
        # print('calculating statistics')
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
        # self.robust_variance_estimate = self.robust_variance_estimator(Y=target.to_numpy(),
        #                                                                 T=treatment.to_numpy(),
        #                                                                 g_pred=g_pred)
        return self

    def calculate_variance_reduction(self):
        """Calculate variance reduction for method

        Returns:
            float: variance reduction rate
        """

        frac = (
            self.regression_results.bse[1] / self.baseline_results.bse[1]
        )  # NOTE: regular, not robust errors

        return 1 - frac**2

    @staticmethod
    def ci_width(ols_results, alpha=0.05):  # TODO: typing
        """Calculate the `1-alpha * 100%` confidence interval."""

        return (
            ols_results.conf_int(alpha=alpha, cols=None)[1][1]
            - ols_results.conf_int(alpha=alpha, cols=None)[0][1]
        )

    def calculate_ci_width_reduction(self, alpha: float = 0.05) -> float:
        """Calculate confidence interval width reduction."""

        frac = self.ci_width(self.regression_results, alpha) / self.ci_width(
            self.baseline_results, alpha
        )

        return 1 - frac
