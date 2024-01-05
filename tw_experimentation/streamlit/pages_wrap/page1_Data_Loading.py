import streamlit as st
from tw_experimentation.streamlit.streamlit_utils import (
    load_data,
    initalise_session_states,
    reset_exp_variables,
    swap_checkbox_state,
    cols_to_select,
    exp_config_to_json,
)
from tw_experimentation.streamlit.streamlit_utils import generate_experiment_output
from tw_experimentation.utils import ExperimentDataset
import json

import copy


def page_1_data_loading():
    st.session_state.update(st.session_state)

    DATA_PAGE = "Data Loading"
    initalise_session_states()

    LOAD_TYPES = ["from .csv", "Snowflake query"]
    SNOWFLAKE_IMPORT = ["query", "table name"]

    st.session_state["last_page"] = DATA_PAGE

    st.header("Data import")
    st.write(
        "If you simply want to use the sample size calculator without any dataset, "
        "navigate to the Experiment Design page."
    )

    load_type = st.radio("Select how you would like to import the data:", LOAD_TYPES)

    if load_type == "from .csv":
        file = st.file_uploader(
            "Upload your data file", type="csv", on_change=reset_exp_variables
        )

        if file is not None:
            df = load_data(file)
            st.session_state["data_loader"].df = df
            st.write("File uploaded")

    if load_type == "Snowflake query":
        restart_snowflake = False
        snowflake_import = st.radio("Import via", SNOWFLAKE_IMPORT)

        if snowflake_import == "query":
            st.text_input("SQL query", key="query")

            snowflake_pull_kwargs = {
                "sql_query": st.session_state["query"],
                "user": st.session_state["snowflake_username"],
                "restart_engine": restart_snowflake,
            }

        if snowflake_import == "table name":
            col11, col12, col13 = st.columns(3)

            with col11:
                st.text_input(
                    "Warehouse",
                    key="warehouse",
                )
            with col12:
                st.text_input(
                    "Schema",
                    key="schema",
                )
            with col13:
                st.text_input("Table", key="table")
            snowflake_pull_kwargs = dict(
                source_database=st.session_state["warehouse"],
                source_schema=st.session_state["schema"],
                source_table=st.session_state["table"],
                user=st.session_state["snowflake_username"],
                restart_engine=restart_snowflake,
            )

        enter_username = False
        if not st.session_state.has_snowflake_connection:
            enter_username = True
        else:
            restart_snowflake = st.checkbox("Restart snowflake connection")

        st.session_state["snowflake_username"] = st.text_input(
            "Snowflake username",
            st.session_state["snowflake_username"],
            disabled=not (enter_username or restart_snowflake),
        )

        if st.button("Fetch data from snowflake"):
            st.session_state["data_loader"].pull_snowflake_table(
                **snowflake_pull_kwargs
            )
            st.session_state.has_snowflake_connection = True

    if st.session_state["data_loader"].df is not None:
        st.divider()
        with st.expander("Load configuration from json"):
            st.write(
                """
                     You can load a previously saved experiment configuration from a json file.
                     In addition, you can load previously already calculated results from a pickle file.
                     """
            )
            config_json = st.file_uploader("Upload a json config file", type="json")
            # output_precalculated = st.file_uploader(
            #     "Upload a pickle file with precalculated results", type="pkl"
            # )

            if config_json is not None:
                try:
                    json_content = config_json.getvalue()
                    exp_config = json.loads(json_content)
                    st.session_state["is_experiment"] = exp_config.get(
                        "is_experiment", True
                    )
                    st.session_state["variant_temp"] = exp_config.get(
                        "variant_name", None
                    )
                    st.session_state["outcomes_temp"] = exp_config.get("outcomes", [])
                    st.session_state["timestamp_temp"] = exp_config.get(
                        "timestamp", None
                    )
                    st.session_state["pre_experiment_temp"] = exp_config.get(
                        "pre_experiment", []
                    )
                    st.session_state["segments_temp"] = exp_config.get("segments", [])
                    st.session_state["is_dynamic_experiment_temp"] = exp_config.get(
                        "is_dynamic_experiment", False
                    )
                    st.session_state["remove_outliers"] = exp_config.get(
                        "remove_outliers", True
                    )
                except json.JSONDecodeError:
                    st.error("Invalid JSON file. Please upload a valid JSON file.")

        st.subheader("Defining the experiment")
        st.session_state["is_experiment"] = st.toggle(
            (
                "On: Data is from an experiment. Off: Data is only pre-experimental for"
                " experiment design"
            ),
            value=(
                st.session_state["is_experiment"]
                if "is_experiment" in st.session_state
                else True
            ),
            on_change=swap_checkbox_state,
            args=("is_experiment",),
            help="""If you have data from an experiment, you need to specify at least outcomes and variant. 
            If you only have pre-experiment data, you only need to specify pre-experiment data metrics.""",
        )
        st.write(
            "Please specify your column(s) with  main outcome metrics, variants and"
            " timestamp"
        )

        cols_to_exclude = [
            st.session_state[f"{var}_temp"]
            for var in [
                "outcomes",
                "variant",
                "timestamp",
                "segments",
                "pre_experiment",
                "is_dynamic_experiment",
            ]
        ]
        cols_for_selection = cols_to_select(
            st.session_state["data_loader"].df.columns, cols_to_exclude
        )

        if st.session_state["is_experiment"]:
            st.selectbox(
                "Column name of variant assignment",
                (
                    list(cols_for_selection) + [st.session_state["variant_temp"]]
                    if st.session_state["variant_temp"] is not None
                    else list(cols_for_selection)
                ),
                index=None,
                key="variant_temp",
            )
            st.markdown(
                ":heavy_exclamation_mark: **Data requirement:** "
                + "Variants should be in the format 0,1,2,.. where 0 is the control"
                " group and 1,[2,...] treatment group(s)"
                + " E.g. you need to transform labels 'Control Group' to 0 and"
                " 'Treatment"
                " Group' to 1 :heavy_exclamation_mark: "
            )

            st.multiselect(
                "Outcomes",
                (
                    list(cols_for_selection) + st.session_state["outcomes_temp"]
                    if st.session_state["outcomes_temp"] is not None
                    else list(cols_for_selection)
                ),
                key="outcomes_temp",
            )
            st.markdown(
                "**Data requirement:** Entries in the outcome columns must be numbers."
            )
            st.session_state["remove_outliers"] = st.toggle(
                "Remove outliers from outcomes",
                value=(
                    st.session_state["remove_outliers"]
                    if "remove_outliers" in st.session_state
                    else True
                ),
                on_change=swap_checkbox_state,
                args=("remove_outliers",),
            )

        st.multiselect(
            "Pre-experiment outcome metrics [if available]",
            (
                list(cols_for_selection) + st.session_state["pre_experiment_temp"]
                if st.session_state["pre_experiment_temp"] is not None
                else list(cols_for_selection)
            ),
            key="pre_experiment_temp",
            help="""The pre-experiment data can be used for two things:
            1. For sample size calculations on the Experiment Design page;
            2. the data is used to increase the sensitivity of an AB test (higher power). 
            This can be applied in the page Experiment Evaluation""",
        )
        if st.session_state["is_experiment"]:
            st.session_state["is_dynamic_experiment_temp"] = st.toggle(
                label="Assignment is dynamic",
                value=st.session_state["is_dynamic_experiment_temp"],
                on_change=swap_checkbox_state,
                args=("is_dynamic_experiment_temp",),
                help="""If the assignment is dynamic, then we need to provide a column with timestamps.
                This is not necessary if all assignment happen at once, e.g. in a marketing campaign.""",
            )
            if st.session_state["is_dynamic_experiment_temp"]:
                st.selectbox(
                    "Column name of timestamp",
                    (
                        list(cols_for_selection) + [st.session_state["timestamp_temp"]]
                        if st.session_state["timestamp_temp"] is not None
                        else list(cols_for_selection)
                    ),
                    index=None,
                    key="timestamp_temp",
                )

            st.multiselect(
                "Segments (for monitoring and segmentation analysis) [if available]",
                (
                    list(cols_for_selection) + st.session_state["segments_temp"]
                    if st.session_state["segments_temp"] is not None
                    else list(cols_for_selection)
                ),
                key="segments_temp",
            )

        st.write(
            'To update your experiment setup, you need to click the "Define data model"'
            " button below"
        )
        if st.button("Define data model"):
            reset_exp_variables(
                vars=[
                    "exp_design_target",
                    "freq_eval_segments",
                    "wp_target",
                    "output_loaded",
                    "bayes_target_plot",
                ]
            )
            for s in ["outcomes", "variant", "timestamp", "segments", "pre_experiment"]:
                st.session_state[s] = copy.deepcopy(st.session_state[s + "_temp"])

            if not st.session_state["is_experiment"]:
                assert (
                    len(st.session_state["pre_experiment"]) > 0
                ), "Must specify at least one pre-experiment metric"
                st.session_state.ed = ExperimentDataset(
                    data=copy.deepcopy(st.session_state["data_loader"].df),
                    variant="",
                    targets="",
                    pre_experiment_cols=st.session_state["pre_experiment"],
                    is_only_pre_experiment=True,
                )
                st.session_state.ed.preprocess_pre_experiment_dataset()
            else:
                if not st.session_state["is_dynamic_experiment_temp"]:
                    st.session_state["timestamp"] = None
                st.session_state.ed = ExperimentDataset(
                    data=copy.deepcopy(st.session_state["data_loader"].df),
                    variant=st.session_state["variant"],
                    targets=st.session_state["outcomes"],
                    date=(
                        st.session_state["timestamp"]
                        if st.session_state["is_dynamic_experiment"]
                        else None
                    ),
                    n_variants=st.session_state["data_loader"]
                    .df[st.session_state["variant"]]
                    .nunique(),
                    pre_experiment_cols=st.session_state["pre_experiment"],
                    segments=st.session_state["segments"],
                )
                st.session_state.ed.preprocess_dataset(
                    remove_outliers=st.session_state["remove_outliers"]
                )
            st.session_state["is_defined_data_model"] = True
        if st.session_state["ed"] is not None:
            st.write(
                """
                     You can also generate the all results at this point. If you do not click the button,
                     the output will be computed as you click through the pages. 
                     Generating the results may take a while and you can decide to do it now or later. 
                     """
            )
            if st.button("Compute results"):
                if not st.session_state["is_experiment"]:
                    st.session_state["output_loaded"] = None
                else:
                    if not st.session_state["is_dynamic_experiment_temp"]:
                        st.session_state["timestamp"] = None
                    st.cache_data.clear()
                    generate_experiment_output(
                        copy.deepcopy(st.session_state["data_loader"].df),
                        variant=st.session_state["variant"],
                        targets=st.session_state["outcomes"],
                        date_created=(
                            st.session_state["timestamp"]
                            if st.session_state["is_dynamic_experiment"]
                            else None
                        ),
                        pre_experiment_cols=st.session_state["pre_experiment"],
                        segments=st.session_state["segments"],
                        remove_outliers=st.session_state["remove_outliers"],
                    )
    # coldict = {
    #     "lightgreen": "outcomes",
    #     "lightblue": "variant",
    #     "lightcoral": "timestamp",
    #     "orange": "pre_experiment",
    #     "violet": "segments",
    # }
    # coldict = {
    #     l: v
    #     for v, k in coldict.items()
    #     for l in st.session_state[k]
    #     if st.session_state[k] is not None
    # }

    # def highlight_cols(s, coldict):
    #     if s.name in coldict.keys():
    #         return ["background-color: {}".format(coldict[s.name])] * len(s)
    #     return [""] * len(s)

    # Currently defined data model
    if st.session_state["is_defined_data_model"] and isinstance(
        st.session_state.ed, ExperimentDataset
    ):
        st.divider()
        st.subheader("Currently selected experiment data:")
        st.dataframe(
            data=st.session_state.ed.data.head(100)
            .style.set_properties(
                subset=st.session_state["outcomes"],
                **{"background-color": "lightgreen"},
            )
            .set_properties(
                subset=st.session_state["variant"],
                **(
                    {"background-color": "lightblue"}
                    if st.session_state.is_experiment
                    else {}
                ),
            )
            # TODO: fix bug with timestamp coloring
            # .set_properties(
            #     subset=st.session_state["timestamp"], **{"background-color": "lightcoral"}
            # )
            .set_properties(
                subset=st.session_state["pre_experiment"],
                **({"background-color": "orange"}),
            )
            .set_properties(
                subset=st.session_state["segments"],
                **(
                    {"background-color": "violet"}
                    if st.session_state.is_experiment
                    else {}
                ),
            )
        )
        st.write(
            "Column highlighting: :green[outcomes], :blue[variant], :red[timestamp],"
            " :orange[pre-experiment], :violet[segments]"
        )

        st.caption("Outcome metrics (with metric type):")
        metric_types = []
        if st.session_state.ed.targets is not None:
            metric_types += st.session_state.ed.targets
        if st.session_state.ed.pre_experiment_cols is not None:
            metric_types += st.session_state.ed.pre_experiment_cols
        st.json(
            {
                k: v
                for k, v in st.session_state.ed.metric_types.items()
                if k in metric_types
            }
        )
        st.caption("Timestamp column:")
        st.write(st.session_state.ed.date)
        st.caption("Variant column:")
        st.write(st.session_state.ed.variant if st.session_state.is_experiment else "")
        st.caption("Segments")
        st.write(st.session_state.ed.segments)

        with st.expander("Download experiment configuration"):
            st.text_input("Experiment name", key="exp_name")
            st.download_button(
                "Download experiment configuration as a .json file",
                exp_config_to_json(),
                f"{st.session_state.exp_name}_config.json",
                "application/json",
            )

    # Always display current state of data model:
    # ExperimentDataset is defined
    # Variant = None or <name>
    # df.head()
