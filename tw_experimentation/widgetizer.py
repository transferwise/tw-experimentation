import pandas as pd
import numpy as np


from tw_experimentation.utils import ExperimentDataset, highlight
from tw_experimentation.statistical_tests import FrequentistTest

from tw_experimentation.setuper import Setuper, effect_size_to_uplift
from tw_experimentation.checker import Monitoring
from tw_experimentation.segmentation_frequentist import Segmentation
from tw_experimentation.plotting.monitoring_plots import (
    fig_variant_segment_dependence,
)
from scipy.stats import chi2_contingency


import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ipywidgets as widgets
from ipywidgets import (
    VBox,
    HBox,
    interact,
    interactive,
    interactive_output,
    Label,
    FloatSlider,
    FloatText,
    Select,
    SelectMultiple,
)
from IPython.display import display

METRIC_TYPE_OPTIONS = [
    "binary",
    # "discrete",
    "continuous",
]
DAY_COL = "day"


class MonitoringInterface:
    def __init__(self, ed) -> None:
        self.ed = ed
        # self.monitor = Monitoring(self.ed)
        # self.monitor._plot_sample_ratio_mismatch()
        # self.monitor.target_monitoring()

        extra_cols = (
            set(self.ed.data.columns)
            - set(self.ed.targets)
            - set([self.ed.date])
            - set([self.ed.variant])
            - set([DAY_COL])
        )
        """
        self.monitor_metric_wt = widgets.SelectMultiple(
            options=extra_cols, description="Outcome"
        )

        @widgets.interact_manual(metrics=self.monitor_metric_wt)
        def plot_extra_metric(metrics):
            metrics = list(metrics)
            self.monitor.sanity_metric_monitoring(metrics)

        self.target_wt = widgets.Select(options=self.ed.targets, description="Outcome")

        self.segment_wt = widgets.Select(options=extra_cols, description="Segment")

        @widgets.interact_manual(target=self.target_wt, segment=self.segment_wt)
        def plot_segment(target, segment):
            self.monitor.segment_monitoring(target, segment)
        """

    # TODO: Delete commented code, if unused
    # def segments_checks(self, segments):
    #     title_wt = widgets.HTML(value="<h2><b>Experiment Monitoring</b></h2>")
    #     display(title_wt)
    #     m = Monitoring(ed=self.ed)
    #     df_dyn = m.dynamic_sample_size_descriptives()

    #     sample_size_today = m.total_sample_size_now()

    #     msg_global = f"<h3> Sample Size</h3>"
    #     for k in range(len(sample_size_today)):
    #         msg_global += f"Variant {k}"
    #         if k == 0:
    #             msg_global += " (Control)"
    #         msg_global += f": {sample_size_today.iloc[k,0]}</br>"

    #     msg_global += f"Total Sample Size: {sample_size_today.sum().iloc[0]}"

    #     chi_squared_global_wt = widgets.HTML(value=msg_global)
    #     display(chi_squared_global_wt)
    #     g = sns.lineplot(
    #         data=df_dyn,
    #         x=self.ed.date,
    #         y="variant_cnt",
    #         hue=self.ed.variant,
    #         palette="dark",
    #     )
    #     plt.xticks(rotation=45)
    #     plt.show()
    #     p_values = {"segment": [], "p-value": []}
    #     chi_squared_plots = []
    #     for segment in segments:
    #         # Chi-squared-test
    #         contingency_table = pd.crosstab(
    #             self.ed.data[self.ed.variant], self.ed.data[segment]
    #         )
    #         chi2, p_value, _, expected = chi2_contingency(contingency_table)
    #         p_values["segment"].append(segment)
    #         p_values["p-value"].append(p_value)
    #         p_values_table = pd.DataFrame.from_dict(p_values)
    #         p_values_table["is significant"] = False
    #         p_values_table["decision"] = (
    #             "Independence hypothesis between group number and segment value cannot"
    #             " be rejected"
    #         )
    #         # plot chi squared visualisation
    #         chi_squared_stat_table = m.chi_squared_table(segment)
    #         fig = fig_variant_segment_dependence(chi_squared_stat_table, self.ed)
    #         chi_squared_plots.append(fig)

    #     p_values_table.loc[p_values_table["p-value"] < 0.05, "is significant"] = True
    #     p_values_table.loc[p_values_table["p-value"] < 0.05, "decision"] = (
    #         "Dependence between group number and segment value (The segment is"
    #         " distributed disproportionally along the groups)"
    #     )
    #     msg = (
    #         f"<h3> Distribution test </h3>The <b>Chi-squared-test</b> that there is no"
    #         f" association or relationship between the variant and the segment in an"
    #         f" A/B test has p-values"
    #     )
    #     display(widgets.HTML(value=msg))
    #     display(
    #         p_values_table.style.set_table_attributes(
    #             "style='display:inline'"
    #         ).applymap(
    #             lambda v: "color:green;" if v is True else "color:red;",
    #             subset=["is significant"],
    #         )
    #     )
    #     # fig = make_subplots(rows=1, cols=len(segments))
    #     # for k in range(len(chi_squared_plots)):
    #     #    fig.add_trace(chi_squared_plots[k].data[0], row=1, col=k + 1)
    #     for fig in chi_squared_plots:
    #         display(fig)

    #     for segment in segments:
    #         msg = f"<h3> {segment} </h3>"
    #         display(widgets.HTML(value=msg))
    #         df_dyn_seg = m.dynamic_sample_size_descriptives(segment=segment)
    #         fig = px.line(
    #             df_dyn_seg,
    #             x=self.ed.date,
    #             y="variant_cnt",
    #             color=self.ed.variant,
    #             facet_row=segment,
    #             width=600,
    #             height=600,
    #         )
    #         fig.show()

    #         fig, axes = plt.subplots(1, figsize=(6, 4))

    #         sns.histplot(
    #             data=self.ed.data,
    #             x=self.ed.data[segment],
    #             hue=self.ed.variant,
    #             palette="bright",
    #             multiple="dodge",
    #             alpha=0.3,
    #         )
    #         fig.tight_layout(pad=2.0)
    #         plt.suptitle("Distribution of " + segment)
    #         fig.show()

    # def segment_check(self, segment=None):
    #     m = Monitoring(ed=self.ed)
    #     plt.show()
    #     if segment is not None:
    #         sample_size_segments = m.total_sample_size_now(segment=segment)
    #         vals_segment = dict(self.ed.data[segment].value_counts().apply(int))
    #
    #         # Chi-squared-test
    #         contingency_table = pd.crosstab(
    #             self.ed.data[self.ed.variant],
    #             self.ed.data[segment]
    #         )
    #         chi2, p_value, _, expected = chi2_contingency(contingency_table)
    #
    #         msg = f"<h1> Segment monitoring </h1> <h3> Distribution test </h3>The <b>Chi-squared-test</b> that there is no association or relationship between the variant and the segment in an A/B test <b>{segment}</b> has p-value"
    #         msg += f"</br> p-value {p_value:.3f}"
    #         msg += "<h3> Sample Size</h3>"
    #         for k in vals_segment.keys():
    #             msg += f"<b>{k}</b>: <br>"
    #             for j in range(len(sample_size_segments.xs(k, level=1))):
    #                 msg += f"Variant {j}"
    #                 if j == 0:
    #                     msg += " (Control)"
    #                 msg += f": {sample_size_segments.xs(k, level=1).iloc[j,0]}</br>"
    #
    #         segment_pvals_wt = widgets.HTML(value=msg, hue=self.ed.variant)
    #         display(segment_pvals_wt)
    #
    #         df_dyn_seg = m.dynamic_sample_size_descriptives(segment=segment)
    #         # g_seg = sns.FacetGrid(
    #         #    df_dyn_seg, row=segment, hue=self.ed.variant, aspect=2
    #         # )
    #         # g_seg.map(sns.lineplot, self.ed.date, "variant_cnt")
    #         # plt.xticks(
    #         #    rotation=45,
    #         #    horizontalalignment="right",
    #         #    fontweight="light",
    #         # )
    #         fig = px.line(
    #             df_dyn_seg,
    #             x=self.ed.date,
    #             y="variant_cnt",
    #             color=self.ed.variant,
    #             facet_row=segment,
    #             width=600,
    #             height=600,
    #         )
    #         fig.show()
    #
    #         fig, axes = plt.subplots(1, figsize=(6, 4))
    #
    #         sns.histplot(
    #             data=self.ed.data,
    #             x=self.ed.data[segment],
    #             hue=self.ed.variant,
    #             palette="bright",
    #             multiple="dodge",
    #             alpha=0.3,
    #         )
    #         fig.tight_layout(pad=2.0)
    #         plt.suptitle("Distribution of " + segment)
    #         plt.show()

    # display(chi_squared_gloabl_wt)

    # return VBox([title_wt] + list(freqs.values()))

    # freq_box = widgets.HBox(list(freqs.values()))
    # target_frequencies = [freqs[j].value for j in range(self.ed.n_variants)]
    # m = Monitoring(ed=self.ed)
    # ss_plot_wt = widgets.interactive_output(m._plot_sample_size, {})
    # chi_squared_wt = widgets.HTML(value=f"Chi-Squared-Test p-value: {p_val:.3f}")
    # ss_numbers_wt = widgets.HTML(value=f"Chi-Squared-Test p-value: {p_val:.3f}")
    # @widgets.interact(**freqs)
    # def global_sample_size(**kwargs):
    #    m = Monitoring(ed=self.ed)
    #    p_val = m._chi_squared_global(f_exp=list(kwargs.values()))
    #    return p_val

    # widgets.interact(global_sample_size, **freqs)
    # global_wt = widgets.HBox([ss_plot_wt, chi_squared_wt])
    # exp_monitor = widgets.VBox([title_wt, freq_box, global_wt])

    # display(exp_monitor)


class SegmentationInterface:
    def __init__(self, ed) -> None:
        self.ed = ed

        extra_cols = (
            set(self.ed.data.columns)
            - set(self.ed.targets)
            - set([self.ed.date])
            - set([self.ed.variant])
            - set([DAY_COL])
        )

        self.segment_choice_wt = widgets.Dropdown(
            options=extra_cols, description="Segment"
        )
        title_wt = widgets.HTML(value="<h2><b>Evaluation: Segmentation</b></h2>")
        display(title_wt)

        @widgets.interact_manual(category=self.segment_choice_wt)
        def manual_segmentation(category):
            s = Segmentation(self.ed)
            df_summary = s.frequentist_by_segment(category)
            display(
                df_summary.style.apply(highlight, axis=1)
                .set_table_attributes("style='display:inline'")
                .bar(subset=["Estimated_Effect_relative"], color="grey")
                .format(precision=3)
            )


class SampleSizeInterface:
    def __init__(self) -> None:
        pass

    def classical_test(self):
        pass
        # treatment share
        # if continuous: sd
        # if binary:
        # baseline conversion
        # uplift relative or absolute?

        self.alpha_widget = widgets.FloatSlider(
            value=0.05,
            min=0,
            max=1.0,
            step=0.01,
            description="Alpha (Type-I error):",
            continuous_update=True,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
            style=dict(description_width="initial"),
        )

        self.beta_widget = widgets.FloatSlider(
            value=0.2,
            min=0,
            max=1.0,
            step=0.01,
            description="Beta (Type-II error):",
            continuous_update=True,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
            style=dict(description_width="initial"),
        )
        self.type_widget = widgets.ToggleButtons(
            options=METRIC_TYPE_OPTIONS,
            description=f"Key outcome metric type:",
            disabled=False,
            style=dict(description_width="initial"),
        )
        self.tshare_widget = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1.0,
            step=0.01,
            description="Treatment Share:",
            disabled=False,
            continuous_update=True,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
            style=dict(description_width="initial"),
        )
        # binary outcome specifications
        self.bc_widget = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1.0,
            step=0.001,
            description="Baseline Conversion:",
            disabled=False,
            continuous_update=True,
            orientation="horizontal",
            readout=True,
            readout_format=".3f",
            style=dict(description_width="initial"),
        )

        self.et_widget = widgets.ToggleButtons(
            options=["absolute", "relative"],
            description=f"Effect type:",
            disabled=False,
            style=dict(description_width="initial"),
        )
        self.es_widget = widgets.FloatText(
            value=0.05,
            description="Uplift to detect:",
            disabled=False,
            readout_format=".3f",
            style=dict(description_width="initial"),
        )

        # continuous outcome specifications
        self.sd_widget = widgets.FloatText(
            value=1,
            description="standard deviation:",
            disabled=False,
            style=dict(description_width="initial"),
        )

        self.mean_widget = widgets.FloatText(
            value=100,
            description="baseline mean:",
            disabled=False,
            style=dict(description_width="initial"),
        )

        self.type_widget.observe(self._continuous_binary_switch, names="value")

        self.sd_widget.layout.visibility = "hidden"
        self.mean_widget.layout.visibility = "hidden"

        title_wt = widgets.HTML(
            value="<h1><b>Sample Size Calculator</b></h1>",
        )
        display(title_wt)
        """        @widgets.interactive(
            alpha=self.alpha_widget,
            beta=self.beta_widget,
            tshare=self.tshare_widget,
            effect_type=self.et_widget,
            effect_size=self.es_widget,
            metric_type=self.type_widget,
            baseline_conversion=self.bc_widget,
            sd=self.sd_widget,
            mean=self.mean_widget,
        )
        """

        def tester(
            alpha,
            beta,
            tshare,
            effect_type,
            effect_size,
            metric_type,
            baseline_conversion,
            mean,
            sd,
        ):
            if metric_type == "binary":
                sd = np.sqrt(baseline_conversion * (1 - baseline_conversion))
                setup = Setuper.from_uplift(
                    alpha=alpha,
                    beta=beta,
                    uplift=effect_size,
                    sd=sd,
                    mean=baseline_conversion,
                    relation=effect_type,
                    treatment_share=tshare,
                )
                sample_size = setup.sample_size_two_sample_proportion_z_test()
                max_sample_size = 2 * sample_size["Total Sample Size"]
                x_sample_size = np.linspace(
                    200 if max_sample_size > 500 else 20, max_sample_size, num=100
                )
                uplift_map = lambda x: effect_size_to_uplift(
                    setup.effect_size_two_sample_z_test(x),
                    baseline_conversion,
                    sd,
                    relation=effect_type,
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

            elif metric_type in ["discrete", "continuous"]:
                setup = Setuper.from_uplift(
                    alpha=alpha,
                    beta=beta,
                    uplift=effect_size,
                    sd=sd,
                    mean=mean,
                    relation=effect_type,
                    treatment_share=tshare,
                )
                sample_size = setup.sample_size_t_test()
                max_sample_size = 2 * sample_size["Total Sample Size"]
                x_sample_size = np.linspace(
                    200 if max_sample_size > 500 else 20, max_sample_size, num=100
                )
                uplift_map = lambda x: effect_size_to_uplift(
                    setup.effect_size_t_test(x),
                    mean,
                    sd,
                    relation=effect_type,
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

            result_wt = widgets.HTML(
                value=(
                    "<h2>Total Sample Size:"
                    f" {sample_size['Treatment Sample Size'] + sample_size['Control Sample Size']}</h2><h3>Treatment"
                    f" Sample Size: {sample_size['Treatment Sample Size']}<br> Control"
                    f" Sample Size: {sample_size['Control Sample Size']}</h3>"
                )
            )
            display(result_wt)
            display(fig)

        wt = widgets.interactive(
            tester,
            alpha=self.alpha_widget,
            beta=self.beta_widget,
            tshare=self.tshare_widget,
            effect_type=self.et_widget,
            effect_size=self.es_widget,
            metric_type=self.type_widget,
            baseline_conversion=self.bc_widget,
            sd=self.sd_widget,
            mean=self.mean_widget,
        )
        display(wt)

    def _continuous_binary_switch(self, change):
        if self.type_widget.value == "binary":
            self.bc_widget.layout.visibility = "visible"
            # self.et_widget.layout.visibility = "visible"
            self.sd_widget.layout.visibility = "hidden"
            self.mean_widget.layout.visibility = "hidden"
        else:
            self.bc_widget.layout.visibility = "hidden"
            # self.et_widget.layout.visibility = "hidden"
            self.sd_widget.layout.visibility = "visible"
            self.mean_widget.layout.visibility = "visible"


class DataWidget:
    def __init__(self, df) -> None:
        self.df = df

        title_wt = widgets.HTML(
            value=(
                "<h2><b>Defining the Data:</b></h2> <h3>Please select the relevant"
                " columns</h3>"
            ),
        )
        display(title_wt)

        self.target_widget = widgets.SelectMultiple(
            options=self.df.columns,
            description="Outcomes",
            style=dict(description_width="initial"),
        )
        display(self.target_widget)

        self._variant_col_name()
        self._date_label()

        self.pre_exp_cols = widgets.SelectMultiple(
            options=self.df.columns,
            description="Pre-Experiment Columns",
            style=dict(description_width="initial"),
        )
        display(self.pre_exp_cols)

        self.mts = {}
        for col in self.df.columns:
            self.mts[col] = widgets.ToggleButtons(
                options=METRIC_TYPE_OPTIONS,
                description=f"Metric Type of {col}",
                disabled=False,
                style=dict(description_width="initial"),
            )
            display(self.mts[col])
            self.mts[col].layout.visibility = "hidden"
            self.mts[col].layout.display = "none"

        self.target_widget.observe(self._target_change, names="value")

    def create_data_model(self):
        dm_button = widgets.Button(description="Create data model!")

        dm_button.on_click(self._create_experiment_dataset)
        display(dm_button)

    def _target_change(self, change):
        for target in set(change.old):
            self.mts[target].layout.visibility = "hidden"

        for target in set(change.new):
            self.mts[target].layout.visibility = "visible"
            self.mts[target].layout.display = "block"

    def _variant_col_name(self):
        self.variant_col = widgets.Dropdown(
            options=self.df.columns,
            description="Name of Column with Variants",
            style=dict(description_width="initial"),
        )
        display(self.variant_col)

    def _date_label(self):
        self.timestamp_col = widgets.Dropdown(
            options=self.df.columns,
            description="Name of Column with Timestamps",
            style=dict(description_width="initial"),
        )
        display(self.timestamp_col)

    def _create_experiment_dataset(self, b):
        self.ed = ExperimentDataset(
            data=self.df,
            variant=self.variant_col.value,
            targets=self.target_widget.value,
            date=self.timestamp_col.value,
            pre_experiment_cols=self.pre_exp_cols.value,
            n_variants=self.df[self.variant_col.value].nunique(),
            metric_types={
                target: self.mts[target].value for target in self.target_widget.value
            },
        )
        self.ed.preprocess_dataset()


class FrequentistEvaluation:
    def __init__(self, ed) -> None:
        self.ed = ed

    def start(self):
        title_wt_freq = widgets.HTML(value="<h2><b>Test Evaluation</b></h2>")
        display(title_wt_freq)

        if self.ed.n_variants > 2:
            # mt_widget = widgets.Dropdown(
            #    options=["No", "Yes"],
            #    description="Multitest correction:",
            #    disabled=False,
            #    value="No",
            # )
            # display(mt_widget)

            interact(
                self._frequentist_process,
                has_correction=widgets.Dropdown(
                    options=["No", "Yes"],
                    value="Yes",
                    description="multi-test correction:",
                    disabled=False,
                    style=dict(description_width="initial"),
                ),
            )
            # self.correction = mt_widget.value
        # run_frequentist_bn = widgets.Button(description="Run hypothesis test!")
        # run_frequentist_bn.on_click(self._frequentist_process)
        # self._frequentist_process()
        else:
            self._frequentist_process(False)

    def _frequentist_process(self, has_correction):
        multitest_correction = None
        if has_correction == "Yes":
            multitest_correction = "bonferroni"
        ft = FrequentistTest(ed=self.ed, multitest_correction=multitest_correction)
        ft.compute()

        self.result_df = ft.get_results_table()
        display(
            self.result_df.style.apply(highlight, axis=1)
            .set_table_attributes("style='display:inline'")
            .bar(subset=["Estimated_Effect_relative"], color="grey")
            .format(precision=3)
        )


# class BayesianEvaluation:
#     def __init__(self, ed) -> None:
#         self.ed = ed
#         self.bt = BayesTest(ed=ed)
#
#     def start(self):
#         DEFAULT_KEY_VARIABLES = {
#             "binary": ["probs"],
#             "continuous": ["loc", "gate"],
#             "discrete": ["rate", "gate"],
#         }
#
#         self.bt.compute_posterior()
#         for target in self.ed.targets:
#             self.bt.plot_posterior(
#                 target=target,
#                 likelihood_variables=DEFAULT_KEY_VARIABLES[
#                     self.ed.metric_types[target]
#                 ],
#             )
#
#             figures = self.bt.plot_posterior_difference(
#                 target=target,
#                 likelihood_variables=DEFAULT_KEY_VARIABLES[
#                     self.ed.metric_types[target]
#                 ],
#             )
