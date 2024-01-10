import sys, os, pathlib


root_path = os.path.realpath("../..")
sys.path.insert(0, root_path)

sys.path.append(str(pathlib.Path().absolute()).split("/tw_experimentation")[0])
import streamlit as st

st.write(sys.path)
import pandas as pd
from scipy.stats import chi2_contingency


### For PullAndMatchData
from tw_experimentation.utils import ExperimentDataset
from tw_experimentation.statistical_tests import (
    FrequentistTest,
    compute_frequentist_results,
    run_cuped,
)
from tw_experimentation.segmentation_frequentist import (
    Segmentation,
    run_segmentation_analysis,
)
from tw_experimentation.setuper import ExpDesignAutoCalculate
from tw_experimentation.result_generator import generate_results

from tw_experimentation.utils import variant_name_map
from tw_experimentation.plotting.monitoring_plots import (
    fig_variant_segment_dependence,
)


from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
import json

from typing import Optional, List, Union

from tw_experimentation.constants import (
    COLORSCALES,
    ACCOUNT,
    REGION,
    USERNAME,
    AUTHENTICATOR,
    DATABASE,
    WAREHOUSE,
    SOURCE_DATABASE,
    SOURCE_SCHEMA,
    SOURCE_TABLE,
    RESULT_DATABASE,
    RESULT_SCHEMA,
    RESULT_TABLE,
    ID_COLUMN,
    TIMESTAMP_COLUMN,
)
from tw_experimentation.checker import (
    Monitoring,
    SegmentMonitoring,
    SequentialTest,
    NormalityChecks,
)
from tw_experimentation.bayes.bayes_test import BayesTest


def fetch_data_from_table_name(warehouse: str, schema: str, table: str):
    st.session_state["sf_table_import"] = "Not implemented yet"


def exp_config_to_json():
    """Converts the current session state to a json file"""
    config = {
        "exp_name": st.session_state.exp_name,
        "is_experiment": st.session_state.is_experiment,
        "variant_name": st.session_state.variant,
        "timestamp": st.session_state.timestamp,
        "is_dynamic_experiment": st.session_state.is_dynamic_experiment,
        "pre_experiment": st.session_state.pre_experiment,
        "segments": st.session_state.segments,
        "outcomes": st.session_state.outcomes,
        "remove_outliers": st.session_state.remove_outliers,
    }
    config_json = json.dumps(config)
    return config_json


def initalise_session_states():
    STATE_VARS = {
        "last_page": "Main",
        "output_loaded": None,
        "warehouse": "WAREHOUSE",
        "schema": "SCHEMA",
        "table": "<MY_TABLE_NAME>",
        "query": "SELECT * FROM database.schema.table",
        "snowflake_username": "<FIRSTNAME.LASTNAME>@SITE.com",
        "has_snowflake_connection": False,
        "outcomes": [],
        "variant": None,
        "is_dynamic_experiment_temp": True,
        "is_dynamic_experiment": True,
        "is_experiment": True,
        "timestamp": None,
        "segments": [],
        "segments_temp": [],
        "outcomes_temp": [],
        "pre_experiment_temp": [],
        "variant_temp": None,
        "timestamp_temp": None,
        "ed": None,
        "data_loader": PullAndMatchData(),
        "is_defined_data_model": False,
        "evaluate_CUPED": False,
        "exp_design_alpha": 5,
        "exp_design_beta": 20,
        "exp_design_treatment_share": 50,
        "exp_design_baseline_conversion": 0.5,
        "exp_design_mde": 0.05,
        "exp_design_sd": 20,
        "exp_design_metric_type": "binary",
        "exp_design_baseline_mean": 100,
        "exp_design_target": None,
        "log_transform_in_target_plots": False,
        "multi-test-correction": "Yes",
        "evaluation_change_alpha": False,
        "evaluation_alpha": 0.05,
        "wp_segmentation_treatment": 1,
        "wp_min_segments": 3,
        "wp_max_depth": 3,
        "bayes_threshold": 0.0,
        "bayes_threshold_lower": 0.0,
        "bayes_threshold_upper": 1.0,
        "bayes_rope_upper": 1.0,
        "bayes_rope_lower": -1.0,
    }
    for k, v in STATE_VARS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_data(path: str):
    df = pd.read_csv(path)
    return df


def reset_exp_variables(vars: Optional[List] = None):
    if vars is None:
        vars = [
            "outcomes_temp",
            "variant_temp",
            "timestamp_temp",
        ]
    for s in vars:
        if s in st.session_state:
            del st.session_state[s]

        initalise_session_states()


def cols_to_select(data_loader_cols: List, cols_to_exclude: List[Union[List, None]]):
    """Helper function to select columns from data loader"""
    cols_to_select = set(data_loader_cols)
    for cols in cols_to_exclude:
        if cols is not None:
            if isinstance(cols, list):
                cols_to_select = cols_to_select - set(cols)
            elif isinstance(cols, str):
                cols_to_select = cols_to_select - set([cols])
    return cols_to_select


class PullAndMatchData:
    """Class for
    - pulling data from snowflake
    - put data into ExperimentDataset instance
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        outcomes: Optional[List] = (None,),
        variant: Optional[str] = (None,),
        target: Optional[str] = None,
        ed: Optional[ExperimentDataset] = None,
    ) -> None:
        self._df = df
        self._outcomes = outcomes
        self._variant = variant
        self._target = target
        self._ed = ed
        self.engine = None
        self.connection = None

    def pull_snowflake_table(
        self,
        sql_query=None,
        user=USERNAME,
        source_database=SOURCE_DATABASE,
        source_schema=SOURCE_SCHEMA,
        source_table=SOURCE_TABLE,
        save_to_csv=False,
        restart_engine=False,
        filename="experiment.csv",
        path="",
    ):
        if self.engine is None or self.connection is None or restart_engine:
            self.engine = create_engine(
                URL(
                    account=ACCOUNT,
                    region=REGION,
                    user=user,
                    authenticator=AUTHENTICATOR,
                    database=DATABASE,
                    warehouse=WAREHOUSE,
                )
            )
            self.connection = self.engine.connect()
        # try:
        if sql_query is None:
            sql_query = f"""
                select * from {source_database}.{source_schema}.{source_table}
                """
        df = pd.read_sql(sql_query, self.connection)
        # finally:
        #    connection.close()
        #    self.engine.dispose()

        if save_to_csv:
            df.to_csv(path + filename)

        self._df = df
        return df

    def data_to_snowflake(
        self,
        user=USERNAME,
        result_database=RESULT_DATABASE,
        result_schema=RESULT_SCHEMA,
        result_table=RESULT_TABLE,
        restart_engine=False,
        if_exists="fail",
    ):
        if self.engine is None or restart_engine:
            self.engine = create_engine(
                URL(
                    account=ACCOUNT,
                    region=REGION,
                    user=user,
                    authenticator=AUTHENTICATOR,
                    database=result_database,
                    warehouse=WAREHOUSE,
                )
            )
            self.connection = self.engine.connect()
        self._df.applymap(str).to_sql(
            result_table,
            self.connection,
            schema=result_schema,
            if_exists=if_exists,
            index=False,
        )
        print("success")
        # connection.close()
        # engine.dispose()

    def define_data_model(
        self,
        data: Optional[Union[pd.DataFrame, ExperimentDataset]] = None,
        variant: Optional[str] = None,
        targets: Optional[List[str]] = None,
        event_timestamp: Optional[str] = None,
        outcomes: Optional[List[str]] = None,
        is_dynamic: Optional[bool] = True,
    ):
        """Create data model / ExperimentDataset with all relevant specification
        If an ExperimentDataset instance is supplied: Still updates based on other inputs such as
        targets, variant etc. if those are supplied

        Args:
            data (Optional[Union[pd.DataFrame, ExperimentDataset]], optional):
                dataframe or ready ExperimentDataset.
                If None, relies on data pulled from Snowflake. Defaults to None.
            variant (Optional[str], optional): Name of variant columnn. Defaults to None.
            targets (Optional[List[str]], optional): List of primary metrics. Defaults to None.
            event_timestamp (Optional[str], optional): Timestamp if available. Defaults to None.
            outcomes (Optional[List[str]], optional): List of outcome metrics.
                Outcomes that are not targets will be excluded from analysis. Defaults to None.
            is_dynamic(Optional[bool], optional): Whether experiment is dynamic or static. Defaults to True
        """

        if variant is not None:
            self.variant = variant
        if targets is not None:
            self._target = targets
        if outcomes is not None:
            self.outcomes = outcomes

        if isinstance(data, ExperimentDataset):
            self._ed = data
            self.df = self._ed.data
        else:
            if data is None:
                assert (
                    self.df is not None
                ), "Data needs to be supplied directly or pulled from Snowflake before"
                data = self.df
            else:
                assert isinstance(data, pd.DataFrame)
                self.df = data
            self._ed = ExperimentDataset(
                data,
                self.variant,
                self.target,
                date=event_timestamp,
                outcomes=self.outcomes,
                is_dynamic=is_dynamic,
            )
        return self._ed

    def _datamodel_is_defined(self):
        if not isinstance(self._ed, ExperimentDataset):
            'An ExperimentDataset needs to be defined through the method "define_data_model()"'
        return isinstance(self._ed, ExperimentDataset)

    @property
    def ed(self):
        assert self._datamodel_is_defined(), "Need to define ExperimentDataset"
        return self._ed

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    @property
    def outcomes(self):
        if self._datamodel_is_defined():
            return self._ed.outcomes
        else:
            return self._outcomes

    @outcomes.setter
    def outcomes(self, value):
        if self._datamodel_is_defined():
            self._ed.outcomes = value
        else:
            self._outcomes = value

    @property
    def target(self):
        if self._datamodel_is_defined():
            return self._ed.targets
        else:
            return self._target

    @target.setter
    def target(self, value):
        assert value in self.outcomes
        if self._datamodel_is_defined():
            self._ed.targets = value
        self._target = value

    @property
    def variant(self):
        if self._datamodel_is_defined():
            return self._ed.variant
        else:
            return self._variant

    @variant.setter
    def variant(self, value):
        if self._datamodel_is_defined():
            self._ed.variant = value
        else:
            self._variant = value


@st.cache_data
def wrapper_auto_calc_pre_experiment(ed: ExperimentDataset):
    edac = ExpDesignAutoCalculate(ed)
    return edac


def generate_experiment_output(*args, **kwargs):
    output = generate_results(*args, **kwargs)
    ingest_loaded_output(output)
    st.session_state["exp_meta_data"] = st.session_state.output_loaded["emd"]


def ingest_loaded_output(output, temp_only=True):
    """Ingests output from a loaded json file"""
    assert isinstance(output, dict)
    st.session_state["output_loaded"] = output
    if not temp_only:
        st.session_state["outcomes"] = output["emd"].targets
        st.session_state["variant"] = output["emd"].variant
        st.session_state["timestamp"] = output["emd"].date
        st.session_state["segments"] = output["emd"].segments
        st.session_state["pre_experiment"] = output["emd"].pre_experiment_cols
    st.session_state["remove_outliers"] = output.get("remove_outliers", True)
    st.session_state["is_experiment"] = not output["emd"].is_only_pre_experiment
    st.session_state["is_dynamic_experiment"] = output["emd"].is_dynamic_observation


def wrap_button_auto_calc_pre_experiment(target):
    edac = wrapper_auto_calc_pre_experiment(st.session_state.ed)
    metric_type = edac.metric_types[target]
    if metric_type in ["continuous", "discrete"]:
        st.session_state["exp_design_metric_type"] = "continuous"
        st.session_state["exp_design_baseline_mean"] = edac.mean(target)
        st.session_state["exp_design_sd"] = edac.sd(target)
    elif metric_type == "binary":
        st.session_state["exp_design_metric_type"] = "binary"
        st.session_state["exp_design_baseline_conversion"] = edac.mean(target)


@st.cache_data
def monitoring(ed, _output_loaded=None):
    if _output_loaded is None:
        monitor = Monitoring(ed)
        monitor_results = monitor.create_tables_and_plots()
        return monitor_results
    else:
        return _output_loaded["monitor_results"]


@st.cache_data
def segment_monitoring(ed, segments, _output_loaded=None, alpha=0.05):
    if _output_loaded is None:
        segment_monitor = SegmentMonitoring(ed, segments)
        segment_monitor_results = segment_monitor.create_tables_and_plots(
            chi_squared_alpha=alpha
        )
        return segment_monitor_results
    else:
        return _output_loaded["segment_monitor_results"]


@st.cache_data
def normality_check(ed, _output_loaded=None, alpha=0.05):
    if _output_loaded is None:
        nc = NormalityChecks(ed)
        nco = nc.create_results(alpha=alpha)
        return nco
    else:
        return _output_loaded["nco"]


@st.cache_data
def sequential_testing(ed, _output_loaded=None):
    if _output_loaded is None:
        sequential_test = SequentialTest(ed)
        sequential_test.sequential_test_results()
        return sequential_test.fig_sequential_test()
    else:
        return _output_loaded["fig_sequential_test"]


@st.cache_data
def monitor_abs_sample_size(ed, segments=None):
    m = Monitoring(ed=ed)
    df_dyn = m.dynamic_sample_size_descriptives()
    sample_size_now = m.total_sample_size_now()
    variant_names = variant_name_map(st.session_state.ed.n_variants)
    sample_size_now.index = sample_size_now.index.map(variant_names)
    SAMPLE_SIZE_TITLE = "Sample size"
    sample_size_now.columns = [SAMPLE_SIZE_TITLE]
    sample_size_total = pd.DataFrame(
        sample_size_now[SAMPLE_SIZE_TITLE].sum(),
        columns=sample_size_now.columns,
        index=["Total sample size"],
    )
    sample_size_now = pd.concat([sample_size_now, sample_size_total])

    p_values = {"segment": [], "p-value": []}
    p_values_table = None
    df_dyn_segments = {}
    if segments is not None:
        chi_squared_heatmaps = {}
        for j, segment in enumerate(segments):
            # Chi-squared-test
            contingency_table = pd.crosstab(ed.data[ed.variant], ed.data[segment])
            chi2, p_value, _, expected = chi2_contingency(contingency_table)
            p_values["segment"].append(segment)
            p_values["p-value"].append(p_value)
            p_values_table = pd.DataFrame.from_dict(p_values)
            p_values_table["is significant"] = False
            p_values_table["decision"] = (
                "Independence hypothesis between group number and segment value cannot"
                " be rejected"
            )

            p_values_table.loc[
                p_values_table["p-value"] < 0.05, "is significant"
            ] = True
            p_values_table.loc[p_values_table["p-value"] < 0.05, "decision"] = (
                "Dependence between group number and segment value (The segment is"
                " distributed disproportionally along the groups)"
            )

            df_dyn_segments[segment] = m.dynamic_sample_size_descriptives(
                segment=segment
            )
            chi_squared_stat_table = m.chi_squared_table(segment)
            chi_squared_heatmaps[segment] = fig_variant_segment_dependence(
                chi_squared_stat_table, ed
            )
            chi_squared_heatmaps[segment].update_layout(
                coloraxis={"colorscale": COLORSCALES[j % len(COLORSCALES)]}
            )

    return (
        sample_size_now,
        df_dyn,
        p_values_table,
        df_dyn_segments,
        chi_squared_heatmaps,
    )


# @st.cache_data
def sequential_test_plot(ed):
    m = Monitoring(ed=ed)
    m.sequential_test_results(ed.targets)
    return m


@st.cache_data
def frequentist_evaluation(ed, has_correction, alpha=0.05):
    multitest_correction = (
        "bonferroni" if ed.n_variants > 2 and has_correction == "Yes" else None
    )
    ft = FrequentistTest(ed=ed, alpha=alpha, multitest_correction=multitest_correction)
    ft.compute()
    return ft.get_results_table()


@st.cache_data
def frequentist_results(ed, _output_loaded=None):
    if _output_loaded is None:
        ftr = compute_frequentist_results(ed)
        return ftr
    else:
        return _output_loaded["frequentist_test_results"]


@st.cache_data
def cuped_results(ed, _output_loaded=None):
    if _output_loaded is None:
        cuped = run_cuped(ed)
        return cuped
    else:
        return _output_loaded["cuped"]


# TODO: Turn caching back on when wise-pizza pickle issues are resolved
# @st.cache_data
def segmentation_analysis(ed, _output_loaded=None):
    if _output_loaded is None:
        segmentation = run_segmentation_analysis(ed, segments=ed.segments)
        return segmentation
    else:
        return _output_loaded["segmentation_analysis"]


# @st.cache_data
def wp_cache_wrapper(ed, target, treatment, segments, max_depth, min_segments):
    segmentation = Segmentation(ed=ed)
    wp_results_table = segmentation.wise_pizza_frequentist(
        target=target,
        treatment=treatment,
        segments=segments,
        max_depth=max_depth,
        min_segments=min_segments,
        auto_display_df=False,
    )

    return wp_results_table, segmentation


@st.cache_data
def bayes_cache_wrapper(ed: ExperimentDataset, _output_loaded=None):
    if _output_loaded is None:
        bt = BayesTest(ed=ed)
        br = bt.compute_posterior()
        return br
    else:
        return _output_loaded["bayes_result"]


def frequentist_segmentation(ed, segments, alpha=0.05):
    s = Segmentation(ed, alpha=alpha)
    df_summary = s.frequentist_by_segment(segments)
    return df_summary


def _coming_from_other_page(current_page, last_page):
    """Check whether coming from other streamlit page

    Args:
        current_page (str): current page name
        last_page (str): page name on last streamlit run

    Returns:
        bool: arriving from other page or not
    """
    return not (current_page == last_page)


def swap_evaluation_change_alpha():
    st.session_state["evaluation_change_alpha"] = not st.session_state[
        "evaluation_change_alpha"
    ]


def swap_checkbox_state(state_label):
    st.session_state[state_label] = not st.session_state[state_label]
