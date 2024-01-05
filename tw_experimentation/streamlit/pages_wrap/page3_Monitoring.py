import streamlit as st

from tw_experimentation.streamlit.streamlit_utils import (
    monitoring,
    segment_monitoring,
    sequential_testing,
    initalise_session_states,
    swap_checkbox_state,
    normality_check,
)


def page_3_monitoring():
    st.session_state.update(st.session_state)

    MONITORING_PAGE = "Monitoring"
    initalise_session_states()

    st.title("Monitoring the experiment")

    if st.session_state["is_defined_data_model"] and st.session_state["is_experiment"]:
        if st.session_state["output_loaded"] is not None:
            emd = st.session_state["output_loaded"]["emd"]
        else:
            emd = st.session_state.ed.experiment_meta_data()

        monitor_results = monitoring(
            st.session_state.ed,
            _output_loaded=st.session_state["output_loaded"],
        )

        col11, col12 = st.columns(2)
        with col11:
            st.dataframe(
                monitor_results.sample_size_table.style.set_table_attributes(
                    "style='display:inline'"
                )
            )
        with col12:
            fig = monitor_results.fig_sample_size_pie
            st.plotly_chart(fig)
        if emd.is_dynamic_observation:
            fig = monitor_results.fig_dynamic_sample_size
            st.plotly_chart(fig)

        if (
            st.session_state["segments"] is not None
            and len(st.session_state["segments"]) > 0
        ):
            ALPHA_CHI_SQUARED = 0.05
            monitor_segments = segment_monitoring(
                st.session_state.ed,
                st.session_state["segments"],
                _output_loaded=st.session_state["output_loaded"],
                alpha=ALPHA_CHI_SQUARED,
            )
            st.header("Distribution test")
            st.write(
                "The Chi-squared-test that there is no association or relationship between"
                " the variant and the segment in an A/B test has p-values"
            )
            st.table(
                monitor_segments.table_chi_squared_test.style.set_table_attributes(
                    "style='display:inline'"
                ).applymap(
                    lambda v: "color:green;" if v is True else "color:red;",
                    subset=[f"is significant at the {ALPHA_CHI_SQUARED} level"],
                )
            )
            segment_independence_explanation = (
                "The following plots visualise the chi-squared test.Darker colors indicate"
                " a higher degree of dependence which can be  warning sign that these pairs"
                " are not assigned correctly.Note that the dependence is just relative to"
                " all other variant-segment value paris per segment. This means not every"
                " dark value indicates a strong dependence, but rather a stronger"
                " dependence than other variant-segment pairs."
            )
            st.subheader(
                "Segment-variant (in)dependence", help=segment_independence_explanation
            )
            # plot chi-squared heatmaps
            for segment in st.session_state["segments"]:
                st.plotly_chart(
                    monitor_segments.figs_chi_squared_heatmaps[segment].update_layout(
                        height=100 * emd.n_variants, width=600
                    )
                )
            st.subheader("Dynamic and static variant per segment counts")
            if emd.is_dynamic_observation:
                for s in st.session_state["segments"]:
                    fig = monitor_segments.figs_segment_sample_size[s]
                    st.plotly_chart(fig)
            for s in st.session_state["segments"]:
                fig = monitor_segments.figs_segment_histograms[s]
                st.plotly_chart(fig)

        st.header("Distribution of outcome metrics")

        # plots for target metrics
        st.session_state["log_transform_in_target_plots"] = st.toggle(
            label=(
                "Use log scale for target metric plots (continuous and discrete metrics"
                " only)"
            ),
            value=st.session_state["log_transform_in_target_plots"],
            on_change=swap_checkbox_state,
            args=("log_transform_in_target_plots",),
        )

        for target in emd.targets:
            fig = monitor_results.fig_target_metric_distribution[target]
            if st.session_state["log_transform_in_target_plots"] and emd.metric_types[
                target
            ] in [
                "discrete",
                "continuous",
            ]:
                fig.update_yaxes(type="log")
            elif not st.session_state["log_transform_in_target_plots"]:
                fig.update_yaxes(type="linear")
            st.plotly_chart(fig)
            if target in monitor_results.fig_target_cdf.keys():
                fig = monitor_results.fig_target_cdf[target]
                if st.session_state["log_transform_in_target_plots"]:
                    fig.update_xaxes(type="log")
                elif not st.session_state["log_transform_in_target_plots"]:
                    fig.update_xaxes(type="linear")
                st.plotly_chart(fig)
            if target in monitor_results.fig_target_qq_variants.keys():
                fig = monitor_results.fig_target_qq_variants[target]
                st.plotly_chart(fig)

        if (
            len(
                [
                    target
                    for target in emd.targets
                    if emd.metric_types[target] in ["continuous", "discrete"]
                ]
            )
            > 0
        ):
            st.divider()
            st.header("Testing for Normality of the data")
            st.markdown(
                """
                When using frequentist tests, we assume that the data is (approximately) normally distributed.
                For continuous and discrete metrics, we can test this assumption using the Q-Q plot and the Shapiro-Wilk test.
                For binary metrics, we do not need to test for normality. 
                """
            )
            st.markdown(
                """
                    **Q-Q plot**: The Q-Q plot compares the quantiles of the data to the quantiles of a normal distribution.
                    If the points in the plot are scattered around the diagonal (blue line), then the data is approximately normally distributed (That's good!)
                        """
            )
            st.markdown(
                """
                **Shapiro-Wilk test**: The Shapiro-Wilk test tests the null hypothesis that the data is normally distributed.
                If there is a significant result to reject the null hypothesis (low p-value), then we using the frequentist test may not be valid."""
            )
            ALPHA_SHAPIRO = 0.05
            nco = normality_check(
                st.session_state.ed,
                _output_loaded=st.session_state["output_loaded"],
                alpha=ALPHA_SHAPIRO,
            )
            for target in nco.targets:
                fig = nco.figs_qqplots[target]
                st.plotly_chart(fig)
                shapiro_df = nco.tables_shapiro_wilk[target]
                st.caption(f"Shapiro-Wilk test for {target}")
                st.dataframe(
                    shapiro_df.style.format(
                        {"p-value": "{:.3f}", "statistic": "{:.3f}"}
                    )
                    .set_table_attributes("style='display:inline'")
                    .applymap(
                        lambda v: "color:green;" if v > ALPHA_SHAPIRO else "color:red;",
                        subset=[f"p-value"],
                    )
                )

        if emd.is_dynamic_observation:
            st.divider()
            st.header("Monitoring treatment effect / Sequential testing over time")
            st.markdown(
                """
            - Here we can check metrics in time, averages and if there were any changes we can see it in the p-value

            Instead of static tests monitoring, you can also run a sequential test which potentially allows you to stop the test early (if the p-value in the following plot is below a predefined threshold such as alpha=.05).

            Below you can see 
            - left column: average of each outcome over time (for both treatments and control)
            - center column: treatment effect with confidence interval over time for each treatment group
            - right column: dynamic p-value based on a sequential hypothesis test for each treatment
            """
            )
            st.session_state["last_page"] = MONITORING_PAGE

            sequential_test_fig = sequential_testing(
                st.session_state.ed, _output_loaded=st.session_state["output_loaded"]
            )
            st.plotly_chart(sequential_test_fig)
    elif not st.session_state["is_experiment"]:
        st.write(
            "You have only provided pre-experiment data. "
            'Please define the experiment in "Experiment Design" first to use Monitoring.'
        )
    else:
        st.write("You first need to Choose or Configure the experiment.")
