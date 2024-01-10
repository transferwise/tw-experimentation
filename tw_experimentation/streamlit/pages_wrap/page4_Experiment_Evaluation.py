import streamlit as st
from tw_experimentation.streamlit.streamlit_utils import (
    _coming_from_other_page,
    swap_evaluation_change_alpha,
    initalise_session_states,
    swap_checkbox_state,
    frequentist_results,
    cuped_results,
    segmentation_analysis,
)

import numpy as np
from wise_pizza.plotting import plot_segments

from tw_experimentation.utils import highlight


def page_4_experiment_evaluation():
    # results

    st.session_state.update(st.session_state)

    CURRENT_PAGE = "Experiment Evaluation"
    initalise_session_states()

    st.header("Experiment evaluation")

    if st.session_state["is_defined_data_model"] and st.session_state["is_experiment"]:
        if st.session_state["output_loaded"] is not None:
            emd = st.session_state["output_loaded"]["emd"]
        else:
            emd = st.session_state.ed.experiment_meta_data()

        frequentist_output = frequentist_results(
            st.session_state.ed, _output_loaded=st.session_state["output_loaded"]
        )

        st.markdown(
            """
            - In this section you can test hypotheses about your groups
            - And check if you have a statistically significant result by getting the p-value
            """
        )

        st.session_state["evaluation_change_alpha"] = st.checkbox(
            "Change significance level (defaults to .05)",
            value=st.session_state["evaluation_change_alpha"],
            on_change=swap_evaluation_change_alpha,
        )

        if st.session_state["evaluation_change_alpha"]:
            st.session_state["evaluation_alpha"] = st.slider(
                r"significance level $ \alpha $ (Type-I error)",
                min_value=0.001,
                max_value=100.0,
                format="%.2f%%",
                # key="evaluation_alpha",
                value=(
                    st.session_state["evaluation_alpha"]
                    if _coming_from_other_page(
                        CURRENT_PAGE, st.session_state["last_page"]
                    )
                    else 5.0
                ),
                step=0.01,
            )
        else:
            st.session_state["evaluation_alpha"] = 5.0
        alpha = st.session_state.evaluation_alpha
        st.markdown(
            """ 
            In case you have multiple treatment variants, you will have the option to perform a multi-test correction on the p-values 
            (current implementation: Bonferroni method. This is subject to change in the future).
            """
        )

        if emd.n_variants > 2:
            st.session_state["multi-test-correction"] = st.radio(
                "Multi-test correction:", ["Yes", "No"]
            )

        st.subheader("Frequentist Evaluation")
        multitest_correction = (
            "bonferroni"
            if emd.n_variants > 2 and st.session_state["multi-test-correction"] == "Yes"
            else None
        )

        st.dataframe(
            frequentist_output.compute_stats_per_target(
                type_i_error=alpha / 100, multitest_correction=multitest_correction
            )
            .get_results_table()
            .style.apply(highlight, axis=1)
            .set_table_attributes("style='display:inline'")
            .bar(subset=["Estimated_Effect_relative"], color="grey")
            .format(precision=3),
        )

        st.markdown(
            """
            #### How can I interpret the results?

            - We compare each treatment variant with the control group
            - The statistic of interest is the difference in means for every specified metric.
            - This yields estimated treatment effects and p-value for them
            - By convention, if p-value < 0.05 the result is regarded significant
            - We also provide confidence intervals for metrics
            """
        )
        st.divider()

        if (
            emd.pre_experiment_cols is not None
            and emd.pre_experiment_cols != []
            and emd.pre_experiment_cols != [None]
        ):  # TODO: could probably be done better
            st.session_state["evaluate_CUPED"] = st.toggle(
                "Power/Sensitivity improvement via CUPED",
                value=(
                    st.session_state["evaluate_CUPED"]
                    if "evaluate_CUPED" in st.session_state
                    else False
                ),
                on_change=swap_checkbox_state,
                args=("evaluate_CUPED",),
                help="""
            CUPED is a variance reduction method leveraging pre-experiment data in order to increase the sensitivity of an A/B test. 
            The basic idea is to use pre-experiment data as a control variate in the test;  the pre-experiment data is used to transform the target variable so that its variability is lowered after which we apply the standard/vanilla T-test to the transformed target.
            """,
            )
            if st.session_state["evaluate_CUPED"]:
                st.markdown(
                    """
                    The columns specified as pre-experiment metrics in Data Loading must be independent of the variant assignment.
                    The greater the correlation between the chosen columns and the outcome, the greater the CUPED's variance reduction and hence power/sensitivity improvement.
                    **The simplest and best option is to use the pre-experiment outcome if such exists**. For example, if the outcome metric is volume of transactions, then the pre-experiment outcome could be the volume of transactions in the month prior to the experiment.
                    """
                )
                st.subheader("Test with power/sensitivity improvement")
                cuped_output = cuped_results(
                    st.session_state.ed,
                    _output_loaded=st.session_state["output_loaded"],
                )

                st.dataframe(
                    cuped_output.results_table(
                        has_correction=(
                            emd.n_variants > 2
                            and st.session_state["multi-test-correction"] == "Yes"
                        ),
                        alpha=alpha / 100,
                    )
                    .style.set_table_attributes(  # .apply(highlight, axis=1)\
                        "style='display:inline'"
                    )  # .bar(subset=["Estimated_Effect_relative"], color="grey")
                    .format(precision=3)
                )

                st.markdown(
                    """
                    #### How can I interpret the results?
                    - The above presents the result of the T-test after the power/sensitivity method has been applied
                    - The conclusions which can be drawn are the same as for the vanilla T-tests

                    #### What if the CUPED's effect significance is different from the Frequentist Evaluation's effect significance?
                    - Generally, we can still consider the effect to be significant if CUPED is significant while Frequentist Evaluation is not.
                    - However, if it's the other way and CUPED's effect is not significant while Frequentist Evaluation's is, we have a reason to believe that there is some issue with CUPED that requires further investigation.
                    """
                )
                st.divider()
        st.subheader("Segmentation with Wise Pizza")
        st.markdown(
            """ 
            - In this section you can find unusual segments in terms of the difference between the control and test groups
            - Please provide segments which you want to analyse, metric to analyse and number of observations
            """
        )

        st.markdown(
            """ 
            Find segments whose average is most different from the global one

            - `segments`: List of discrete dimensions to find slices
            - `target`: Metric to analyse
            - `treatment`: If you have different test groups, specify group here, for example treatment=1 means 
            we compare with first treatment group
            - `min_segments`: Minimum number of segments to find
            - `max_depth`: Maximum number of dimension to constrain in segment definition
                
                
            *Warning*: The p-values are currently not corrected for multiple comparisons. 
            However, Wise-Pizza identifies segments as interesting only if the treatment effect is sufficiently high 
            compared to the segment sample size so this selection is a first approximation of avoiding 
            p-value inflation in segmentation analysis.
            """
        )

        if (
            st.session_state["segments"] is None
            or len(st.session_state["segments"]) == 0
        ):
            st.write("No segments available for wise pizza segmentation analysis")
        else:
            segmentation_output = segmentation_analysis(
                st.session_state.ed, _output_loaded=st.session_state["output_loaded"]
            )
            if (
                st.session_state["segments"] is not None
                or len(st.session_state.segments) > 0
            ):
                if emd.n_variants > 2:
                    st.session_state.wp_segmentation_treatment = st.radio(
                        "Treatment to analyse",
                        np.arange(emd.n_variants - 1) + 1,
                    )
                else:
                    st.session_state.wp_segmentation_treatment = 1
                st.selectbox("Target", emd.targets, key="wp_target")

                st.dataframe(
                    segmentation_output.wise_pizza_output(
                        st.session_state.wp_target,
                        st.session_state.wp_segmentation_treatment,
                    )
                    .style.apply(highlight, axis=1)
                    .set_table_attributes("style='display:inline'")
                    .bar(subset=["Estimated_Effect_relative"], color="grey")
                    .format(precision=3)
                )
                wp_fig = segmentation_output.wise_pizza_figs[
                    st.session_state.wp_target
                ][st.session_state.wp_segmentation_treatment]
                st.plotly_chart(wp_fig, use_container_width=True)
                st.write("Expand figure below to fullscreen for best view.")

                st.markdown(
                    """ 
                    #### How can I interpret the results?

                    - We are trying to find unusual segments in terms of the averages (***to highlight the segments contributing the most to the difference between test and control***)
                    - Impact (Blue) is the model coefficient * size, if it is bigger than zero, than segment average is bigger than the global one (global difference in metric between test and control)
                    - Simple segment averages (Red) - average for specific segment
                    - Segment Sizes (Green) - number if observations in the segment

                    """
                )
            else:
                st.write("Select segments to analyse.")

            st.divider()
            st.subheader("Evaluation for segments")
            st.write(
                "In this section you can get p-value and statistical resuls for the"
                " selected segment"
            )

        if (
            st.session_state["segments"] is None
            or len(st.session_state["segments"]) == 0
        ):
            st.write("No segments available for frequentist segmentation analysis")
        else:
            st.markdown(
                """ 
                #### How can I interpret the results?

                - For each variant we compare it with the control group

                - Here we have mean values for every specified metric.
                - Then we have estimated effects and p-value for them
                - By default if p-value < 0.05 the result is significant
                - And then we have confidence intervals for metrics
                """
            )
            st.selectbox(
                "Segment to analyse",
                st.session_state["segments"],
                key="freq_eval_segments",
            )
            if len(st.session_state.freq_eval_segments) > 0:
                s = (
                    segmentation_output.segment_output(
                        st.session_state.freq_eval_segments, alpha=alpha / 100
                    )
                    .reset_index(level=[st.session_state.freq_eval_segments])
                    .style
                    # .apply(highlight, axis=1)
                    .set_table_attributes("style='display:inline'")
                    # .bar(subset=["Estimated_Effect_relative"], color="grey")
                    .format(precision=3)
                )
                st.dataframe(s)
    elif not st.session_state["is_experiment"]:
        st.write(
            "You have only provided pre-experiment data. "
            'Please define the experiment in "Experiment Design" first to use Experiment Evaluation.'
        )

    else:
        st.write("You first need to Choose or Configure the experiment..")

    st.session_state["last_page"] = CURRENT_PAGE
