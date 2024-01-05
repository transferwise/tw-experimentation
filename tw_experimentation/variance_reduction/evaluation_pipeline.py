from tw_experimentation.variance_reduction.variance_reduction_method import (
    VarianceReductionMethod,
)
from tw_experimentation.variance_reduction.utils import (
    subsample_data,
)
import pandas as pd
from typing import List
from typing_extensions import Self
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import copy


class VREvaluation:
    """Implements a class for evaluating variance reduction methods."""

    def __init__(self, method: VarianceReductionMethod):
        self.method = method

        # store experiment parameters
        self.treatment_column = None
        self.target_column = None
        self.covariate_columns = None
        self.model = None
        self.model_init_config = None
        self.model_fit_config = None
        self.bootstrap_samples = None
        self.true_ate = None

        # store estimators / instances of variance reduction methods
        self.estimators = []  # NOTE: commented to improve memory efficiency

        # store statistics
        self.estimates = []
        self.p_values = []
        self.biases = []
        self.cis = []
        self.ci_coverages = []
        self.variance_reduction_rates = []
        self.ci_width_reduction_rates = []

        # store baseline statistics
        self.baseline_estimates = []
        self.baseline_p_values = []

    def run(
        self,
        data: pd.DataFrame = None,
        treatment_column: str = None,
        target_column: str = None,
        true_ate: float = None,
        method_params_map={},
        verbose: bool = False,
        bootstrap_samples: np.array = None,
        n_bootstrap: int = 1000,
    ) -> Self:
        """Run the evaluation.

        Args:
            data (pd.DataFrame, optional): experiment data. Defaults to None.
            treatment_column (str, optional): name of the column containing treatment assignment. Defaults to None.
            target_column (str, optional): name of the column containing target metric. Defaults to None.
            true_ate (float, optional): true average treatment effect. Defaults to None.
            method_params_map (dict, optional): map from a method to a . Defaults to {}.
            verbose (bool, optional): _description_. Defaults to False.
            bootstrap_samples (np.array, optional): _description_. Defaults to None.
            n_bootstrap (int, optional): _description_. Defaults to 1000.

        Returns:
            Self: self
        """

        # validate input
        # TODO

        # store experiment parameters
        self.treatment_column = treatment_column
        self.target_column = target_column
        self.method_params_map = method_params_map
        self.bootstrap_samples = bootstrap_samples
        self.true_ate = true_ate

        # store data summary
        # TODO

        # bootstrap data if not passed in
        if bootstrap_samples is None:
            self.bootstrap_samples = subsample_data(
                data=data, n_bootstrap=n_bootstrap, treatment_column=treatment_column
            )

        if verbose:
            print(f"Running {self.method.__name__}...")

        for bootstrap_indices in tqdm(self.bootstrap_samples, disable=not verbose):
            # fit the estimator
            estimator = self.method()
            estimator = estimator.fit(
                data=data.iloc[bootstrap_indices],
                treatment_column=treatment_column,
                target_column=target_column,
                **method_params_map[self.method.__name__],
            )

            # calculate the statistics of interest
            # self.estimators.append(estimator) # NOTE: commented to improve memory efficiency
            self.estimates.append(estimator.estimate)
            self.cis.append(estimator.conf_int_95)
            self.p_values.append(estimator.p_value)

            self.baseline_estimates.append(estimator.baseline_estimate)
            self.baseline_p_values.append(estimator.baseline_p_value)

            if true_ate is not None:
                self.biases.append(estimator.estimate - true_ate)
                self.ci_coverages.append(
                    true_ate >= estimator.conf_int_95[0]
                    and true_ate <= estimator.conf_int_95[1]
                )

        return self

    def report(self) -> pd.DataFrame:
        """Generate a summary table for the experiment."""

        # mean estimate with 95% CI
        mean_estimate = np.mean(self.estimates)
        mean_estimate_ci = (
            np.quantile(self.estimates, 0.025),
            np.quantile(self.estimates, 0.975),
        )

        if self.true_ate is not None:
            # mean bias with 95% CI
            mean_bias = np.mean(self.biases)
            mean_bias_ci = (
                np.quantile(self.biases, 0.025),
                np.quantile(self.biases, 0.975),
            )

            # coverage probability
            coverage_probability = np.mean(self.ci_coverages)

        else:
            mean_bias = None
            mean_bias_ci = None
            coverage_probability = None

        # mean p_value with 95% CI
        if None not in self.p_values:
            mean_p_value = np.mean(self.p_values)
            mean_p_value_ci = (
                np.quantile(self.p_values, 0.025),
                np.quantile(self.p_values, 0.975),
            )
        else:
            mean_p_value = None
            mean_p_value_ci = None

        # variance reduction
        bootstrapped_var_reduction = 1 - (
            np.var(self.estimates) / np.var(self.baseline_estimates)
        )

        # prepare data summary
        # TODO:

        # prepare the tables
        table = pd.DataFrame(
            {
                "Method": [self.method.__name__],
                "Estimate": [mean_estimate],
                "Estimate CI": [mean_estimate_ci],
                "Bias": [mean_bias],
                "Bias CI": [mean_bias_ci],
                "p-value": [mean_p_value],
                "p-value CI": [mean_p_value_ci],
                "Coverage Probability": [coverage_probability],
                "Variance Reduction": [bootstrapped_var_reduction],
            }
        )

        # store the table in self
        # TODO

        # return the table'd information
        return table

    def plot(self, plot_what: List[str]):
        """Plot the results of the experiment."""

        # plot the distribution of the estimates overlaid with the baseline estimate
        if plot_what == "p-values":  # TODO: convert to a sns distplot
            if None in self.p_values:
                print("p-values not available for this method")
                return

            fig, ax = plt.subplots()
            ax.hist(
                self.p_values,
                bins=20,
                alpha=0.5,
                label=f"{self.method.__name__}",
                color="blue",
            )
            ax.hist(
                self.baseline_p_values, bins=20, alpha=0.5, label="DiM", color="red"
            )
            ax.set_xlabel("p-value")
            ax.set_ylabel("Frequency")
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fancybox=True,
                shadow=True,
                ncol=1,
            )
            ax.set_title(f"p-values for {self.method.__name__}")
            plt.show()

        # plot the distribution of the estimates overlaid with the baseline estimate
        elif plot_what == "estimates":  # TODO: convert to a sns distplot
            # fig, ax = plt.subplots()

            # # Plot histograms with kde=True using seaborn's histplot
            # sns.histplot(self.estimates, bins=20, color='blue', alpha=0.2, label=f'{self.method.__name__}', kde=True, ax=ax)
            # sns.histplot(self.baseline_estimates, bins=20, color='red', alpha=0.2, label='DiM', kde=True, ax=ax)

            # ax.set_xlabel('Estimate')
            # ax.set_ylabel('Density')

            # if self.true_ate is not None:
            #     ax.axvline(x=self.true_ate, color='black', label='True ATE', linestyle='--')

            # ax.legend()
            # ax.set_title(f'Estimates for {self.method.__name__}')
            # plt.show()

            fig, ax = plt.subplots()

            # plot a kde plot of estimates
            sns.kdeplot(
                self.baseline_estimates, fill=False, label="DiM", ax=ax, linewidth=2
            )
            sns.kdeplot(
                self.estimates,
                fill=False,
                label=f"{self.method.__name__}",
                ax=ax,
                linewidth=2,
            )  # TODO: potentially improve bandwidth parameters, also maybe clip

            ax.set_xlabel("Estimate")
            ax.set_ylabel("Density")

            if self.true_ate is not None:
                ax.axvline(
                    x=self.true_ate, color="black", label="True ATE", linestyle="--"
                )

            ax.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fancybox=True,
                shadow=True,
                ncol=1,
            )
            ax.set_title(f"Estimates for {self.method.__name__}")
            plt.show()

        else:
            raise NotImplementedError

        # return plots
        return fig, ax


# ====================================================================================================
# ====================================================================================================
# ====================================================================================================


class VREvaluationAll:
    """Implements a class for evaluating an array of variance reduction methods."""

    def __init__(self, methods: List[VarianceReductionMethod]):
        self.methods = methods

        self.evaluations = {}

        # store all experiment data
        self.data = None
        self.treatment_column = None
        self.target_column = None
        self.method_params_map = None
        self.bootstrap_samples = None
        self.n_bootstrap = None

    def run_all(
        self,
        data: pd.DataFrame,
        treatment_column: str,
        target_column: str,
        method_params_map: dict,
        true_ate: float = None,
        verbose: bool = False,
        bootstrap_samples: np.array = None,
        n_bootstrap: int = 1000,
    ) -> Self:
        """Run the evaluation for all methods.

        Args:
            data (pd.DataFrame): experiment data
            treatment_column (str): name of the column containing the treatment flags
            target_column (str): name of the column containing the target metric
            method_params_map (dict): map from a method to a dict of parameters
            true_ate (float, optional): true average treatment effect. Defaults to None.
            verbose (bool, optional): flag specifying whether to print progress. Defaults to False.
            bootstrap_samples (np.array, optional): an array of bootstrap indices to be used in evaluation. Defaults to None.
            n_bootstrap (int, optional): number of samples to be bootstrapped. Defaults to 1000.

        Returns:
            Self: self
        """

        # save experiment parameters
        # self.data = data
        self.treatment_column = treatment_column
        self.target_column = target_column
        self.method_params_map = method_params_map
        self.bootstrap_samples = bootstrap_samples
        self.n_bootstrap = n_bootstrap
        self.true_ate = true_ate

        # bootstrap data if not passed in
        if self.bootstrap_samples is None:
            self.bootstrap_samples = subsample_data(
                data=data, n_bootstrap=n_bootstrap, treatment_column=treatment_column
            )

        # run evaluations for all methods
        for method in self.methods:
            # instantiate evaluation for a method
            exp = VREvaluation(method)

            # run experiment for a method
            exp.run(
                data=data,
                treatment_column=treatment_column,
                target_column=target_column,
                method_params_map=method_params_map,
                true_ate=true_ate,
                verbose=verbose,
                bootstrap_samples=self.bootstrap_samples,
                n_bootstrap=n_bootstrap,
            )

            # store experiment results
            self.evaluations[method.__name__] = exp

        if verbose:
            print("Done!")

        return self

    def report(self) -> pd.DataFrame:
        """Generate a summary table for the experiment."""

        # prepare table
        table = pd.concat(
            [
                self.evaluations[method_name].report()
                for method_name in self.evaluations.keys()
            ]
        )
        table = table.reset_index(drop=True)

        # store table in self
        # TODO

        return table

    def plot(
        self, plot_what: str, ax=None, show_plot=True, show_legend=True, show_title=True
    ):
        """Plot the results of the experiment."""

        # plot the distribution of p-values for all methods
        if plot_what == "p-values":  # TODO: convert to an sns distplot
            if ax is None:
                fig, ax = plt.subplots()

            for method_name in self.evaluations.keys():
                if None in self.evaluations[method_name].p_values:
                    continue
                ax.hist(
                    self.evaluations[method_name].p_values,
                    bins=20,
                    alpha=0.5,
                    label=f"{method_name}",
                    density=True,
                )
            ax.set_xlabel("p-value")
            ax.set_ylabel("Frequency")
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fancybox=True,
                shadow=True,
                ncol=1,
            )
            ax.set_title(f"p-values for all methods")
            plt.show()

        # plot the distribution of estimates for all methods
        elif plot_what == "estimates":
            # fig, ax = plt.subplots()

            # for method_name in self.evaluations.keys():
            #     # if method_name =='DoublyRobustEstimator':
            #     #     continue
            #     estimates = self.evaluations[method_name].estimates
            #     sns.histplot(estimates, bins=20, alpha=0.1, label=f'{method_name}', kde=True, ax=ax)

            # ax.set_xlabel('Estimate')
            # ax.set_ylabel('Density')

            # if self.true_ate is not None:
            #     ax.axvline(x=self.true_ate, color='black', label='True ATE', linestyle='--')

            # ax.legend()
            # ax.set_title('Estimates for all methods')
            # plt.show()

            if ax is None:
                fig, ax = plt.subplots()

            for method_name in self.evaluations.keys():
                # if method_name == 'MultivariateRegressionAdjusted' or method_name == 'MultivariateRegression':
                #     continue
                estimates = self.evaluations[method_name].estimates
                sns.kdeplot(
                    estimates, fill=False, label=f"{method_name}", ax=ax, linewidth=2
                )  # TODO: potentially improve bandwidth parameters, also maybe clip

            ax.set_xlabel("Estimate")
            ax.set_ylabel("Density")

            if self.true_ate is not None:
                ax.axvline(
                    x=self.true_ate, color="black", label="True ATE", linestyle="--"
                )

            # show the plot only from x=-0.5 to x=0.5
            # ax.set_xlim(-0.5, 0.5)

            if show_legend:
                ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    fancybox=True,
                    shadow=True,
                    ncol=1,
                )
            if show_title:
                ax.set_title("Estimates for all methods")
            if show_plot:
                plt.show()

        else:
            raise NotImplementedError

        return ax


# ====================================================================================================
# ====================================================================================================
# ====================================================================================================


class VREvaluationGrid:
    """Implments a class for running evaluations of all methods while changing their parameters."""

    def __init__(self, methods: List[VarianceReductionMethod]):
        self.methods = methods
        self.evaluation_grid = None

    def run(
        self,
        data: pd.DataFrame,
        treatment_column: str,
        target_column: str,
        method_params_grid: np.ndarray,
        n_bootstrap: int = 100,
        bootstrap_samples: np.array = None,
        true_ate: float = None,
        verbose: bool = True,
    ) -> Self:
        """Run the evaluation for all methods and all parameter sets."""

        # save experiment parameters
        # self.data = data
        self.treatment_column = treatment_column
        self.target_column = target_column
        self.method_params_grid = method_params_grid
        self.bootstrap_samples = bootstrap_samples
        self.n_bootstrap = n_bootstrap
        self.true_ate = true_ate

        # initialize empty evaluation grid
        self.evaluation_grid = np.ndarray(method_params_grid.shape, dtype=object)

        # bootstrap data if not passed in
        if self.bootstrap_samples is None:
            self.bootstrap_samples = subsample_data(
                data=data, n_bootstrap=n_bootstrap, treatment_column=treatment_column
            )

        # run evaluations for all sets of parameters
        for i in np.ndindex(method_params_grid.shape):
            print(f"----- Running evaluation for parameter grid index {i}... -----")
            method_params_map = method_params_grid[i]

            # instantiate evaluation for a method
            exp = VREvaluationAll(self.methods)

            # run experiment for a method
            exp.run_all(
                data=data,
                treatment_column=treatment_column,
                target_column=target_column,
                method_params_map=method_params_map,
                true_ate=true_ate,
                verbose=verbose,
                bootstrap_samples=self.bootstrap_samples,
                n_bootstrap=n_bootstrap,
            )

            # store experiment results
            self.evaluation_grid[i] = exp

        return self

    @staticmethod
    def generate_parameters_grid(
        base_methods_params_map,
        covariate_columns_list,
        models,
        model_init_configs,
        model_fit_configs,
    ):
        """Generate a grid of method-parameters map from the base method-parameters map and a list of covariate columns as well as models."""

        # initialize empty grid
        params_maps_grid = np.ndarray(
            [len(models), len(covariate_columns_list)], dtype=object
        )

        # iterate over all sets of model specs
        for i, (model, model_init_config, model_fit_config) in enumerate(
            zip(models, model_init_configs, model_fit_configs)
        ):
            # iterate over all seets of covariate columns
            for j, covariate_columns in enumerate(covariate_columns_list):
                method_params_map = copy.deepcopy(base_methods_params_map)

                # overwrite appropriate parameters in the base method-parameters map
                for method, method_params_dict in base_methods_params_map.items():
                    if "covariate_columns" in method_params_dict:
                        method_params_map[method][
                            "covariate_columns"
                        ] = covariate_columns

                    if "model" in method_params_dict:
                        method_params_map[method]["model"] = model
                        method_params_map[method][
                            "model_init_config"
                        ] = model_init_config
                        method_params_map[method]["model_fit_config"] = model_fit_config

                    else:
                        pass

                params_maps_grid[i, j] = method_params_map

        return params_maps_grid

    def plot_grid(self):
        """Plot the estimates in a grid."""

        fig, axes = plt.subplots(
            self.evaluation_grid.shape[0],
            self.evaluation_grid.shape[1],
            sharex=True,
            figsize=(
                3 * self.evaluation_grid.shape[1],
                3 * self.evaluation_grid.shape[0],
            ),
        )

        if self.evaluation_grid.shape[0] == 1:
            axes = axes.reshape(1, -1)
        if self.evaluation_grid.shape[1] == 1:
            axes = axes.reshape(-1, 1)

        for i in range(self.evaluation_grid.shape[0]):
            for j in range(self.evaluation_grid.shape[1]):
                self.evaluation_grid[i, j].plot(
                    plot_what="estimates",
                    ax=axes[i, j],
                    show_plot=False,
                    show_legend=False,
                    show_title=False,
                )

        # self.evaluation_grid[0, 1].plot(plot_what='estimates', ax=axes[0, 1], show_plot=False)
        # self.evaluation_grid[1, 1].plot(plot_what='estimates', ax=axes[1, 1], show_plot=False)
        axes[0, -1].legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fancybox=True,
            shadow=True,
            ncol=1,
        )
        plt.show()

    def plot(self, plot_what: str):
        raise NotImplementedError

        if plot_what == "bias_variance_tradeoff":
            assert self.true_ate is not None, "True ATE is not provided."

            # create a grid of reports
            report_grid = np.ndarray(self.evaluation_grid.shape, dtype=object)

            for i in np.ndindex(self.evaluation_grid.shape):
                report_grid[i] = self.evaluation_grid[i].report()

            # create grids of biases, vr and estimates
            bias_grid = np.ndarray(self.evaluation_grid.shape, dtype=float)
            vr_grid = np.ndarray(self.evaluation_grid.shape, dtype=float)
            estimate_grid = np.ndarray(self.evaluation_grid.shape, dtype=float)

            for i in np.ndindex(self.evaluation_grid.shape):
                bias_grid[i] = report_grid[i]["Bias"]
                vr_grid[i] = report_grid[i]["Variance Reduction"]
                estimate_grid[i] = report_grid[i]["Estimate"]

            # plot biases for each method
            fig, axes = plt.subplots(3, 1, figsize=(15, 5))
