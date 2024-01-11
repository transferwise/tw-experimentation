import pytest

from tw_experimentation.data_generation import RevenueConversion
from tw_experimentation.result_generator import (
    generate_results,
    save_output,
    load_output,
)
import os

from plotly import graph_objects as go


@pytest.fixture()
def experiment_df(request):
    rc = RevenueConversion()
    df = rc.generate_data()

    return df


@pytest.fixture()
def experiment_df_with_pre_experiment_data(request):
    rc = RevenueConversion()
    df = rc.generate_data()
    df["pre_experiment_data"] = df["revenue"] * 0.5

    return df


@pytest.mark.parametrize(
    "df", ["experiment_df", "experiment_df_with_pre_experiment_data"]
)
def test_generate_and_dump_output(df, request):
    df = request.getfixturevalue(df)
    """checks if output can be generated and saved"""
    targets = ["conversion", "revenue", "num_actions"]
    output = generate_results(
        df=df, variant="T", targets=targets, date_created="trigger_dates"
    )
    path = os.path.dirname(os.path.realpath(__file__))
    file_name = "_output"
    save_output(path, file_name, output)


def test_dump_and_load_output(experiment_df):
    df = experiment_df
    targets = ["conversion", "revenue", "num_actions"]
    output_temp = generate_results(
        df=df, variant="T", targets=targets, date_created="trigger_dates"
    )
    path = os.path.dirname(os.path.realpath(__file__))
    file_name = "_output"
    save_output(path, file_name, output_temp)
    output = load_output(path, file_name)

    fig_seq = output["fig_sequential_test"]

    assert isinstance(fig_seq, go.Figure)

    fig_br = output["bayes_result"].fig_posterior_difference_by_target("revenue")
    assert isinstance(fig_br, go.Figure)


if __name__ == "__main__":
    pytest.main([__file__])
