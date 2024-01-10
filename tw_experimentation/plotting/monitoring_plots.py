import pandas as pd
import numpy as np
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from functools import wraps

from tw_experimentation.utils import (
    variant_name_map,
    ExperimentDataset,
    variant_color_map,
    variantname_color_map,
    hex_to_rgb,
)
from typing import List, Dict, Union, Optional

import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy.stats.distributions import chi2

MAX_N_POINTS = 8000


# TODO: Potentially replace by https://github.com/predict-idlab/plotly-resampler/
def plotly_reduce_n_points_per_trace(
    fig: go.Figure, max_n_points: int = 5000, min_n_points_per_trace: int = 200
):
    """
    Reduces the number of data points in a Plotly figure to improve memory usage.

    Args:
        fig (plotly.graph_objs.Figure): The Plotly figure to be modified.
        max_n_points (int, optional): The maximum number of data points to keep in the figure. Defaults to 5000.

    Returns:
        plotly.graph_objs.Figure: The modified Plotly figure with reduced data points.
    """
    traces = fig.data
    n_traces = len(traces)
    max_points_per_trace = int(max_n_points / n_traces)
    for trace in traces:
        trace_length = len(trace.x)
        step = max(1, trace_length // max(min_n_points_per_trace, max_points_per_trace))
        selected_indices = list(
            range(0, trace_length, step)
        )  # Include the first element

        # Ensure the last element is included
        if selected_indices[-1] != trace_length - 1:
            selected_indices.append(trace_length - 1)
        if (
            hasattr(trace, "x")
            and hasattr(trace, "y")
            and trace.x is not None
            and trace.y is not None
            and trace.mode != "lines"
        ):
            trace.x = trace.x[selected_indices]
            trace.y = trace.y[selected_indices]

    return fig


def plotly_light_memory(max_n_points=5000, min_n_points_per_trace=200):
    """
    Decorator that reduces the number of points in a Plotly figure to optimize memory usage.

    Args:
        max_n_points (int, optional): The maximum number of points allowed in the figure. Defaults to 5000.
        min_n_points_per_trace (int, optional): The minimum number of points allowed per trace in the figure. Defaults to 200.

    Returns:
        function: The decorated function that returns a Plotly figure with reduced number of points per trace.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            fig = func(*args, **kwargs)
            return plotly_reduce_n_points_per_trace(
                fig,
                max_n_points=max_n_points,
                min_n_points_per_trace=min_n_points_per_trace,
            )

        return wrapper

    return decorator


def fig_variant_segment_dependence(
    chi_squared_table: pd.DataFrame,
    ed: ExperimentDataset,
    text_auto=False,
):
    """Heatmap for chi-squared test of dependence between variant and segment
    Normalises the color by setting the midpoint of the colorscale to
    the 95% quantile of the chi-squared distribution divided by degrees of freedom.
    Hence, a cell achieves above a heat at the colorscale midpoint if all
    cells had the same value, then the test has a p-value of .05.

    Args:
        chi_squared_table (pd.DataFrame): table of N(0,1) distributed statistics
        ed (ExperimentDataset): ExperimentDataset object containing experiment data
        color_index (int, optional): index of color scale to use. Defaults to 1.
        text_auto (bool, optional): whether to display cell values. Defaults to False.

    Returns:
        plotly.graph_objs._figure.Figure: plotly figure
    """
    INVERSE_CDF_THRESHOLD = 0.95
    deg_of_fr = (chi_squared_table.shape[0] - 1) * (chi_squared_table.shape[1] - 1)
    norm_constant = chi2.ppf(INVERSE_CDF_THRESHOLD, deg_of_fr)
    fig = px.imshow(
        chi_squared_table,
        title=(
            " Degree of observed dependence of segment"
            f" {chi_squared_table.axes[1].name} and variant pairs"
        ),
        text_auto=text_auto,
        color_continuous_scale="Reds",
        color_continuous_midpoint=norm_constant / deg_of_fr,
    )

    fig = fig.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(ed.n_variants)),
            ticktext=list(variant_name_map(ed.n_variants).values()),
        )
    )
    # fig.update_layout(
    #     coloraxis={"colorscale": c.COLORSCALES[color_index % len(c.COLORSCALES)]}
    # )
    return fig


def plot_sample_size_pie(ed: ExperimentDataset):
    """Plots a pie chart showing the sample size per variant in an ExperimentDataset.

    Args:
        ed (ExperimentDataset): An ExperimentDataset object containing the data to be plotted.

    Returns:
        fig: A plotly figure object representing the pie chart.
    """
    data = pd.DataFrame(
        {
            "Variant": variant_name_map(ed.n_variants).values(),
            "Value": list(ed.sample_sizes.values()),
        }
    )
    color_map = variantname_color_map(list(variant_name_map(ed.n_variants).values()))
    fig = px.pie(
        data,
        names="Variant",
        values="Value",
        title="Sample size per variant",
        color="Variant",
        color_discrete_map=color_map,
    )
    fig.update_traces(textinfo="percent+value")
    return fig


@plotly_light_memory(max_n_points=MAX_N_POINTS)
def plot_dynamic_sample_size(df_dynamic_sample: pd.DataFrame, ed: ExperimentDataset):
    """
    Plots the sample size per variant over time.

    Args:
        df_dynamic_sample (pd.DataFrame): A DataFrame containing the sample size data.
        ed (ExperimentDataset): An ExperimentDataset object containing information about the experiment.

    Returns:
        A plotly.graph_objs._figure.Figure Plot with sample size evolution.
    """
    fig = px.line(
        df_dynamic_sample,
        x=ed.date,
        y="variant_cnt",
        color=ed.variant,
        color_discrete_sequence=variant_color_map(ed.n_variants),
        title="Sample size per variant over time",
    )
    variant_names = variant_name_map(ed.n_variants)
    fig.for_each_trace(lambda trace: trace.update(name=variant_names[int(trace.name)]))
    return fig


@plotly_light_memory(max_n_points=MAX_N_POINTS)
def target_metric_distribution(
    ed: ExperimentDataset, target: str, use_log: bool = False
):
    fig = go.Figure()
    for i in range(ed.n_variants):
        fig.add_trace(
            go.Histogram(
                x=ed.data.loc[
                    ed.data[ed.variant] == i,
                    target,
                ],
                name=variant_name_map(ed.n_variants)[i],
                marker=dict(color=variant_color_map(ed.n_variants)[i]),
            ),
        )
    fig.update_layout(
        title_text="Distribution of " + target + " per variant",
        xaxis_title_text=target,
        yaxis_title_text="Count",
    )

    if ed.metric_types[target] in ["continuous", "discrete"]:
        for perc in [80, 95, 99]:
            percentile = np.percentile(
                ed.data[ed.data[ed.variant] == 0][target],
                perc,
            )
            fig.add_vline(
                x=percentile,
                line_dash="dash",
                line_color="grey",
                annotation=dict(text=f"{perc}th percentile"),
                annotation_position="top right",
            )
        if use_log:
            fig.update_layout(yaxis_type="log")
    return fig


@plotly_light_memory(max_n_points=MAX_N_POINTS)
def plot_target_metric_cdf(ed: ExperimentDataset, target: str, use_log: bool = False):
    """
    Plots the cumulative density function (CDF) for a given target variable.
    """
    fig = px.ecdf(
        ed.data,
        x=target,
        color=ed.variant,
        title="Cumulative density function (CDF)",
        color_discrete_map=variant_color_map(ed.n_variants),
        log_x=(
            use_log if ed.metric_types[target] in ["continuous", "discrete"] else False
        ),
    )
    fig.for_each_trace(
        lambda trace: trace.update(
            name=variant_name_map(ed.n_variants)[int(trace.name)]
        )
    )
    return fig


@plotly_light_memory(max_n_points=MAX_N_POINTS)
def plot_qq_variants(
    qq_variants: Dict[str, np.ndarray], ed: ExperimentDataset, target: str
) -> go.Figure:
    """
    Plots a Q-Q plot for a given target between different variants.

    Args:
        qq_variants (Dict[str, np.ndarray]): A dictionary containing the quantiles  for each variant.
        ed (ExperimentDataset): An ExperimentDataset object containing the data for the experiment.
        target (str): The target variable to plot.

    Returns:
        go.Figure: A Plotly figure object containing the Q-Q plot.
    """
    # TODO: Add identity line
    color_per_variant = variant_color_map(ed.n_variants)
    fig = go.Figure()
    for variant in range(1, ed.n_variants):
        fig.add_trace(
            go.Scatter(
                x=qq_variants[variant], y=qq_variants[0], mode="markers", name=variant
            )
        )
    fig.for_each_trace(
        lambda trace: trace.update(
            name=variant_name_map(ed.n_variants)[int(trace.name)],
            marker=dict(color=color_per_variant[int(trace.name)]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ed.data[target],
            y=ed.data[target],
            mode="lines",
            line=dict(color="#636efa"),
            name="Identity Line",
        )
    )

    fig.update_layout(
        title=f"Quantile-quantile plot for {target} between variants",
        xaxis_title=f"Control",
        yaxis_title=f"Treatment",
    )
    return fig


@plotly_light_memory(max_n_points=MAX_N_POINTS)
def plot_segment_sample_size(
    ed: ExperimentDataset,
    df_dyn_segments: pd.DataFrame,
    segment: List[str],
):
    color_per_variant = variant_color_map(ed.n_variants)
    fig = px.line(
        df_dyn_segments,
        x=ed.date,
        y="variant_cnt",
        color=ed.variant,
        facet_row=segment,
        title=f"sample size per category in segment {segment}",
        color_discrete_map=color_per_variant,
    )
    fig.for_each_trace(
        lambda trace: trace.update(
            name=variant_name_map(ed.n_variants)[int(trace.name)]
        )
    )
    return fig


def plot_segment_histograms(
    ed: ExperimentDataset,
    segment: List[str],
    biggest_segments: Optional[Union[None, List]] = None,
):
    color_per_variant = variant_color_map(ed.n_variants)
    variant_names = variant_name_map(ed.n_variants)

    if biggest_segments is not None:
        data = ed.data[ed.data[segment].isin(biggest_segments)]
    else:
        data = ed.data
    fig = go.Figure()

    for i in range(ed.n_variants):
        fig.add_trace(
            go.Histogram(
                x=data.loc[
                    data[ed.variant] == i,
                    segment,
                ],
                name=variant_names[i],
                marker=dict(color=color_per_variant[i]),
            ),
        )
    fig.update_layout(
        title_text="Distribution of " + segment + " per variant",
        xaxis_title_text=segment,
        yaxis_title_text="Count",
    )
    return fig


@plotly_light_memory(max_n_points=MAX_N_POINTS)
def plot_sequential_test(ed: ExperimentDataset, df_dyn_avg: pd.DataFrame):
    n_rows = len(ed.targets)

    variant_names = variant_name_map(ed.n_variants)
    color_per_variant = variant_color_map(ed.n_variants)

    fig = make_subplots(
        rows=n_rows,
        cols=3,
        row_titles=ed.targets,
        column_titles=["Metric average", "Treatment Effect", "p Value"],
    )

    for j in range(n_rows):
        fig_aux1 = px.line(
            df_dyn_avg,
            x=ed.date,
            y=f"{ed.targets[j]}_avg",
            color=ed.variant,
            color_discrete_map=color_per_variant,
        )
        fig_aux1.for_each_trace(
            lambda trace: trace.update(name=variant_names[int(trace.name)])
        )
        if j > 0:
            fig_aux1.update_traces(showlegend=False)
        for k in range(ed.n_variants):
            fig.add_trace(fig_aux1.data[k], row=j + 1, col=1)

        fig_aux2 = px.line(
            df_dyn_avg[df_dyn_avg[ed.variant] != 0],
            x=ed.date,
            y=f"{ed.targets[j]}_diff",
            color=ed.variant,
            color_discrete_map=color_per_variant,
        )
        fig_aux2.update_traces(showlegend=False)
        for k in range(1, ed.n_variants):
            fig.add_trace(fig_aux2.data[k - 1], row=j + 1, col=2)
            df_variant = df_dyn_avg[df_dyn_avg[ed.variant] == k]
            for bound in ["lower", "upper"]:
                fig.add_trace(
                    go.Scatter(
                        x=df_variant[ed.date],
                        y=df_variant[f"{ed.targets[j]}_CI_{bound}"],
                        fill="tonexty" if bound == "upper" else None,
                        showlegend=False,
                        line=dict(color=color_per_variant[k], width=0.3),
                        fillcolor=(
                            f"rgba{(*hex_to_rgb(color_per_variant[k][1:]), .2)}"
                            if bound == "upper"
                            else None
                        ),
                    ),
                    row=j + 1,
                    col=2,
                )

        fig_aux3 = px.line(
            df_dyn_avg[df_dyn_avg[ed.variant] != 0],
            x=ed.date,
            y=f"{ed.targets[j]}_p_values",
            color=ed.variant,
            color_discrete_map=color_per_variant,
        )
        fig_aux3.update_traces(showlegend=False)
        # fig_aux3.update_yaxes(range=[0, 1])
        for k in range(ed.n_variants - 1):
            fig.add_trace(fig_aux3.data[k], row=j + 1, col=3)

        fig.update_layout(
            title_text="Outcome metric monitor and sequential testing",
        )
        # fig.update_layout({"xaxis3": {"range": [0, 1.01]}})
    return fig


@plotly_light_memory(max_n_points=MAX_N_POINTS)
def plot_qq_normal(
    ed: ExperimentDataset,
    target: str,
    standardized_residuals: Dict[str, Dict[int, np.ndarray]],
) -> go.Figure:
    """
    Plots a quantile-quantile plot for a given target variable and its standardized residuals for each variant in the experiment.

    Args:
        ed (ExperimentDataset): The experiment dataset containing the data for each variant.
        target (str): The name of the target variable to plot.
        standardized_residuals (Dict[str, Dict[int, np.ndarray]]): A dictionary containing the standardized residuals for each variant and target variable.

    Returns:
        go.Figure: A Plotly figure object containing the quantile-quantile plot.
    """
    plt.ioff()
    fig = go.Figure()
    xdata = []
    ydata = []
    color_per_variant = variant_color_map(ed.n_variants)
    variant_names = variant_name_map(ed.n_variants)
    for variant in range(1, ed.n_variants):
        fig_mpl = sm.qqplot(standardized_residuals[target][variant], line="s")
        qqplot_data = fig_mpl.gca().lines
        xdata.append(qqplot_data[0].get_xdata())
        ydata.append(qqplot_data[0].get_ydata())

        fig.add_trace(
            {
                "type": "scatter",
                "x": qqplot_data[1].get_xdata(),
                "y": qqplot_data[0].get_ydata(),
                "mode": "markers",
                "name": variant_names[variant],
                "marker": dict(color=color_per_variant[variant]),
            }
        )
    all_data = xdata + ydata
    data_concat = np.concatenate(all_data)
    fig.add_trace(
        {
            "name": "identity line",
            "type": "scatter",
            "x": [data_concat.min(), data_concat.max()],
            "y": [data_concat.min(), data_concat.max()],
            "mode": "lines",
            "line": {"color": "#636efa"},
        }
    )

    fig["layout"].update(
        {
            "title": f"Quantile-Quantile Plot: {target}",
            "xaxis": {"title": "Theoretical Quantiles", "zeroline": False},
            "yaxis": {"title": "Sample Quantiles"},
        }
    )
    return fig
