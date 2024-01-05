import streamlit as st
import numpy as np
import plotly.graph_objects as go

from tw_experimentation.setuper import (
    Setuper,
    effect_size_to_uplift,
    ExpDesignAutoCalculate,
)
from tw_experimentation.streamlit.streamlit_utils import (
    initalise_session_states,
    wrapper_auto_calc_pre_experiment,
    wrap_button_auto_calc_pre_experiment,
)


def page_2_experiment_design():
    st.session_state.update(st.session_state)

    EXPERIMENT_PAGE = "Experiment Design"
    initalise_session_states()

    st.session_state["last_page"] = EXPERIMENT_PAGE

    st.header("Sample Size Calculation")

    st.markdown(
        r"""
        We first determine the targeted sample size based on the minimal detectable effect. 

        If we want to document a certain uplift, we also need to provide 
        - an estimate of the population mean (e.g., baseline conversion for binary outcomes),
        - treatment share,
        - type-I error $\alpha$, and
        - type-II error $\beta$. (***$\beta$ = 1 - Power***)
        - effect relation to base metric / mean: 'absolute' or 'relative'
        
        
        """
    )
    st.write(
        "The sample size calculation is based on a two-sample proportion z-test for binary"
        " metrics and"
        + "two-sample t-test with equal variance for continuous outcome metrics."
    )

    targets_pre_experiment = (
        st.session_state.ed.pre_experiment_cols
        if st.session_state.ed is not None
        else None
    )
    if targets_pre_experiment is not None and len(targets_pre_experiment) > 0:
        edac = wrapper_auto_calc_pre_experiment(st.session_state.ed)
        st.divider()
        with st.expander("Auto-calculate from pre-experiment data"):
            st.markdown(
                r"""
                If you have pre-experiment data, you can use it to estimate the targeted sample size.
                The method will auto-fill
                - the target metric type,
                - the baseline value, and
                - the standard deviation (for continuous metrics).
                """
            )
            st.selectbox(
                "Target metric",
                st.session_state["pre_experiment"],
                key="exp_design_target",
            )

            if st.session_state["exp_design_target"] is not None:
                st.button(
                    "Auto_calculate",
                    on_click=wrap_button_auto_calc_pre_experiment,
                    args=(st.session_state["exp_design_target"],),
                )

    st.divider()

    col11, col12 = st.columns(2)

    with col11:
        st.number_input(
            r"$\alpha$ (Type-I error, e.g., 1.0 = 1%)",
            min_value=0.0,
            max_value=100.0,
            step=0.01,
            key="exp_design_alpha",
            format="%.2f",
            )

    with col12:
        st.number_input(
            r"$\beta$ (Type-II error, e.g., 1.0 = 1%)",
            min_value=0.0,
            max_value=100.0,
            step=0.01,
            key="exp_design_beta",
            format="%.2f",
            )

    st.number_input(
        "Treatment share (e.g., 50.0 = 50%)",
        min_value=0.0,
        max_value=100.0,
        step=0.01,
        key="exp_design_treatment_share",
        format="%.2f",
    )

    EFFECT_TYPES = ["relative", "absolute"]

    col21, col22 = st.columns(2)

    with col21:
        st.selectbox("Effect type", EFFECT_TYPES, key="exp_design_effect_type")

    with col22:
        mde_max_val = None
        relative_effect = st.session_state["exp_design_effect_type"] == "relative"
        if relative_effect:
            mde_max_val = 1.0

        st.number_input(
            (
                "Uplift to detect"
                if not relative_effect
                else "Relative uplift to detect (e.g., 0.1 = 10%)"
            ),
            min_value=0.001,
            max_value=mde_max_val,
            step=0.001,
            key="exp_design_mde",
            format="%.3f",
        )

    METRIC_TYPE = ["binary", "continuous"]
    st.selectbox("Key outcome metric type", METRIC_TYPE, key="exp_design_metric_type")

    if st.session_state["exp_design_metric_type"] == "binary":
        st.number_input(
            "Baseline value",
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            key="exp_design_baseline_conversion",
            format="%.3f",
        )
    elif st.session_state["exp_design_metric_type"] == "continuous":
        st.number_input("Baseline mean", min_value=0.0, key="exp_design_baseline_mean")
        st.number_input("Standard deviation", min_value=0.0, key="exp_design_sd")

    ###########################
    ##### Plotting ############
    ###########################

    if st.session_state["exp_design_metric_type"] == "binary":
        sd = np.sqrt(
            st.session_state["exp_design_baseline_conversion"]
            * (1 - st.session_state["exp_design_baseline_conversion"])
        )
        setup = Setuper.from_uplift(
            alpha=st.session_state["exp_design_alpha"] / 100,
            beta=st.session_state["exp_design_beta"] / 100,
            uplift=st.session_state["exp_design_mde"],
            sd=sd,
            mean=st.session_state["exp_design_baseline_conversion"],
            relation=st.session_state["exp_design_effect_type"],
            treatment_share=st.session_state["exp_design_treatment_share"] / 100,
        )
        sample_size = setup.sample_size_two_sample_proportion_z_test()
        max_sample_size = 2 * sample_size["Total Sample Size"]
        x_sample_size = np.linspace(
            200 if max_sample_size > 500 else 20, max_sample_size, num=100
        )
        uplift_map = lambda x: effect_size_to_uplift(
            setup.effect_size_two_sample_z_test(x),
            st.session_state["exp_design_baseline_conversion"],
            sd,
            relation=st.session_state["exp_design_effect_type"],
        )
        uplift = np.array(list(map(uplift_map, x_sample_size)))

        fig = go.FigureWidget(data=go.Scatter(x=x_sample_size, y=uplift))
        fig.add_vline(
            x=sample_size["Total Sample Size"],
            line_width=3,
            line_dash="dash",
            line_color="green",
        )
        fig.update_xaxes(title_text="Sample Size")
        fig.update_yaxes(title_text="Minimum Detectable Uplift")

    elif st.session_state["exp_design_metric_type"] == "continuous":
        setup = Setuper.from_uplift(
            alpha=st.session_state["exp_design_alpha"] / 100,
            beta=st.session_state["exp_design_beta"] / 100,
            uplift=st.session_state["exp_design_mde"],
            sd=st.session_state["exp_design_sd"],
            mean=st.session_state["exp_design_baseline_mean"],
            relation=st.session_state["exp_design_effect_type"],
            treatment_share=st.session_state["exp_design_treatment_share"] / 100,
        )
        sample_size = setup.sample_size_t_test()
        max_sample_size = 2 * sample_size["Total Sample Size"]
        x_sample_size = np.linspace(
            200 if max_sample_size > 500 else 20, max_sample_size, num=100
        )
        uplift_map = lambda x: effect_size_to_uplift(
            setup.effect_size_t_test(x),
            st.session_state["exp_design_baseline_mean"],
            st.session_state["exp_design_sd"],
            relation=st.session_state["exp_design_effect_type"],
        )
        uplift = np.array(list(map(uplift_map, x_sample_size)))
        fig = go.FigureWidget(data=go.Scatter(x=x_sample_size, y=uplift))
        fig.add_vline(
            x=sample_size["Total Sample Size"],
            line_width=3,
            line_dash="dash",
            line_color="green",
        )
        fig.update_xaxes(title_text="Sample Size")
        fig.update_yaxes(title_text="Minimum Detectable Uplift")

    col_ss_1, col_ss_2, col_ss_3 = st.columns(3)

    with col_ss_1:
        st.metric(
            "Total Sample Size",
            value=sample_size["Treatment Sample Size"]
            + sample_size["Control Sample Size"],
        )

    with col_ss_2:
        st.metric(
            "Treatment Sample Size",
            value=sample_size["Treatment Sample Size"],
        )

    with col_ss_3:
        st.metric(
            "Control Sample Size",
            value=sample_size["Control Sample Size"],
        )

    st.plotly_chart(fig)
