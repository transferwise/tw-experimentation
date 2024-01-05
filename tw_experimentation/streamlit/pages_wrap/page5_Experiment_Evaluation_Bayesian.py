import streamlit as st
from tw_experimentation.streamlit.streamlit_utils import (
    bayes_cache_wrapper,
    initalise_session_states,
)
import plotly.graph_objects as go

import numpy as np


def page_5_experiment_evaluation_bayesian():
    st.session_state.update(st.session_state)

    CURRENT_PAGE = "Bayesian Evaluation"
    initalise_session_states()

    st.header("Bayesian AB test evaluation")

    if st.session_state["is_defined_data_model"] and st.session_state["is_experiment"]:
        if st.session_state["output_loaded"] is not None:
            emd = st.session_state["output_loaded"]["emd"]
        else:
            emd = st.session_state.ed.experiment_meta_data()
        st.write(
            "In this section, we fit a Bayesian model on the data.We then use the posterior"
            " distribution to compute the probability that the variant is better than the"
            " control."
        )

        br = bayes_cache_wrapper(
            st.session_state.ed, _output_loaded=st.session_state["output_loaded"]
        )

        st.selectbox("Target", emd.targets, key="bayes_target_plot")

        fig = br.fig_posterior_difference_by_target(
            st.session_state["bayes_target_plot"]
        )
        st.plotly_chart(fig)

        fig = br.fig_posterior_cdf_by_target(st.session_state["bayes_target_plot"])

        st.plotly_chart(fig)

        # TODO: Enable bayes factor when tested properly
        # st.subheader("Bayes Factor")

        description = (
            "The Bayes Factor is the p-value analogue in Bayesian hypothesis testing. "
            "It allows us to compare the hypothesis of no effect (null hypothesis) to the"
            " hypothesis of that there is an effect (alternative hypothesis)."
            "The decision is made based on the Risk, which is the probability that the"
            "the null hypothesis is true given the data (probability of a false discovery)."
        )
        # TODO: Enable bayes factor when tested properly
        # st.write(description)

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=[
                            "<b>Bayes Factor</b>",
                            "<b>Risk (%)</b>",
                            "<b>Decision</b>",
                        ],
                        align="left",
                    ),
                    cells=dict(
                        values=[
                            [
                                f'{br.bayes_factor(st.session_state["bayes_target_plot"], var):.2f}'
                                for var in range(1, st.session_state.ed.n_variants)
                            ],
                            [
                                f'{br.false_discovery_rate(st.session_state["bayes_target_plot"], var)*100:.2f}'
                                for var in range(1, st.session_state.ed.n_variants)
                            ],
                            [
                                br.bayes_factor_decision(
                                    st.session_state["bayes_target_plot"], var
                                )
                                for var in range(1, st.session_state.ed.n_variants)
                            ],
                        ],
                        align="left",
                    ),
                )
            ]
        )
        # TODO: Enable bayes factor when tested properly
        # st.plotly_chart(fig)
        st.subheader("Probability that treatment effect greater or smaller than 0")

        n_variants = emd.n_variants
        n_cols_max = 3
        row_order = n_variants <= n_cols_max + 1

        description = (
            "The probability that the average treatment effect is **greater than 0** for"
            f" outcome metric **{st.session_state['bayes_target_plot']}** is\n"
        )
        st.write(description)

        if row_order:
            cols = st.columns(n_variants - 1)

        for variant in range(1, n_variants):
            if row_order:
                with cols[variant - 1]:
                    st.metric(
                        f"{emd.variant_names[variant]}",
                        value=(
                            f"{br.prob_greater_than_zero(st.session_state['bayes_target_plot'])[variant]*100:.2f} %"
                        ),
                    )
            else:
                st.metric(
                    f"{emd.variant_names[variant]}",
                    value=(
                        f"{br.prob_greater_than_zero(st.session_state['bayes_target_plot'])[variant]*100:.2f} %"
                    ),
                )

        description = (
            "The probability that the average treatment effect is **smaller than 0** for"
            f" outcome metric **{st.session_state['bayes_target_plot']}** is\n"
        )
        st.write(description)

        if row_order:
            cols = st.columns(n_variants - 1)

        for variant in range(1, n_variants):
            if row_order:
                with cols[variant - 1]:
                    st.metric(
                        f"{emd.variant_names[variant]}",
                        value=(
                            f"{(1-br.prob_greater_than_zero(st.session_state['bayes_target_plot'])[variant])*100:.2f} %"
                        ),
                    )
            else:
                st.metric(
                    f"{emd.variant_names[variant]}",
                    value=(
                        f"{(1-br.prob_greater_than_zero(st.session_state['bayes_target_plot'])[variant])*100:.2f} %"
                    ),
                )

        st.subheader(
            "Probability that treatment effect is above or below a custom threshold"
        )
        st.write(
            """

            The probability that the average treatment effect is greater or smaller than a custom `threshold`:
            """
        )

        st.number_input(
            "threshold",
            key="bayes_threshold",
        )

        description = (
            "The probability that the average treatment effect is greater than"
            f" **{st.session_state['bayes_threshold']}** for outcome metric"
            f" **{st.session_state['bayes_target_plot']}** is\n"
        )
        st.write(description)

        if row_order:
            cols2 = st.columns(n_variants - 1)

        for variant in range(1, n_variants):
            if row_order:
                with cols2[variant - 1]:
                    st.metric(
                        f"{emd.variant_names[variant]}",
                        value=(
                            f"{br.prob_greater_than_z(st.session_state['bayes_threshold'], st.session_state['bayes_target_plot'])[variant]*100:.2f} %"
                        ),
                    )
            else:
                st.metric(
                    f"{emd.variant_names[variant]}",
                    value=(
                        f"{br.prob_greater_than_z(st.session_state['bayes_threshold'], st.session_state['bayes_target_plot'])[variant]*100:.2f} %"
                    ),
                )

            description = (
                "The probability that the average treatment effect is smaller than"
                f" **{st.session_state['bayes_threshold']}** for outcome metric"
                f" **{st.session_state['bayes_target_plot']}** is\n"
            )
        st.write(description)

        if row_order:
            cols2 = st.columns(n_variants - 1)

        for variant in range(1, n_variants):
            if row_order:
                with cols2[variant - 1]:
                    st.metric(
                        f"{emd.variant_names[variant]}",
                        value=(
                            f"{br.prob_smaller_than_z(st.session_state['bayes_threshold'], st.session_state['bayes_target_plot'])[variant]*100:.2f} %"
                        ),
                    )
            else:
                st.metric(
                    f"{emd.variant_names[variant]}",
                    value=(
                        f"{br.prob_smaller_than_z(st.session_state['bayes_threshold'], st.session_state['bayes_target_plot'])[variant]*100:.2f} %"
                    ),
                )

        st.subheader(
            "Probability that treatment effect is in ROPE (Region of Practical Equivalence)"
        )

        st.markdown(
            """

            As an analogue to frequentist testing, you can define a region of practical equivalence (ROPE). This is an interval where we assume that when the effect is within the interval, the effect is negligible (e.g. not big enough to make an impact, cover cost of rolling out the change). 

            Another reason for defining a ROPE interval is that when we simply look at the probability that the effect is greater than 0, that probability will be 50% even without using any data if we assume that the effect is centered around 0. 
            
            The ROPE interval sizes are autodetected based on the variance of the outcome metric.
            """
        )
        probs, rope_lower, rope_upper = br.rope(st.session_state["bayes_target_plot"])
        description = (
            "The probability that the average treatment effect is outside the **region of"
            f" practical equivalence ({rope_lower:.2f}, {rope_upper:.2f})** for outcome"
            f" metric **{st.session_state['bayes_target_plot']}** is\n"
        )
        st.write(description)

        if row_order:
            cols3 = st.columns(n_variants - 1)

        for variant in range(1, emd.n_variants):
            if row_order:
                with cols3[variant - 1]:
                    st.metric(
                        f"{emd.variant_names[variant]}",
                        value=f"{probs[variant]*100:.2f} %",
                    )
            else:
                st.metric(
                    f"{emd.variant_names[variant]}",
                    value=f"{probs[variant]*100:.2f} %",
                )

        st.subheader("Probability that treatment effect is outside a custom interval")

        col_input_lower, col_input_upper = st.columns(2)
        with col_input_lower:
            st.number_input(
                "Lower threshold",
                key="bayes_threshold_lower",
            )
        with col_input_upper:
            st.number_input(
                "Upper threshold",
                key="bayes_threshold_upper",
            )

        description = (
            "The probability that the average treatment effect is **outside the interval"
            f" ({st.session_state['bayes_threshold_lower']:.2f},"
            f" {st.session_state['bayes_threshold_upper']:.2f})** for outcome metric"
            f" **{st.session_state['bayes_target_plot']}** is\n"
        )
        st.write(description)

        if row_order:
            cols4 = st.columns(n_variants - 1)

        probs = br.prob_outside_interval(
            st.session_state["bayes_threshold_lower"],
            st.session_state["bayes_threshold_upper"],
            st.session_state["bayes_target_plot"],
        )

        for variant in range(1, emd.n_variants):
            if row_order:
                with cols4[variant - 1]:
                    st.metric(
                        f"{emd.variant_names[variant]}",
                        value=f"{probs[variant]*100:.2f} %",
                    )
            else:
                st.metric(
                    f"{emd.variant_names[variant]}",
                    value=f"{probs[variant]*100:.2f} %",
                )
    elif not st.session_state["is_experiment"]:
        st.write(
            "You have only provided pre-experiment data. "
            'Please define the experiment in "Experiment Design" first to use Experiment Evaluation Bayesian.'
        )

    else:
        st.write("You first need to Choose or Configure the experiment.")

    st.session_state["last_page"] = CURRENT_PAGE
