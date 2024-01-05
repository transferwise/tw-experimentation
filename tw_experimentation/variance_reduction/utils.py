from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
from prettytable import PrettyTable
import pandas as pd
import numpy as np
from typing import List, Generator


def create_pretty_table(
    rows: List[str], columns: List[str], contents: np.ndarray
) -> PrettyTable:  # TODO: verify typing
    """Return a pretty table from given row names, column names and contents."""

    # Create a PrettyTable object
    table = PrettyTable()
    table.field_names = [""] + columns

    # Populate the table with data
    for i, row in enumerate(rows):
        table.add_row([row] + list(contents[i]))

    return table


def query_snowflake(
    query: str,
    username: str,
    warehouse: str,
    database: str,
    account: str = "account_name",
    region: str = "region_name",
) -> pd.DataFrame:
    """Query a snowflake database."""

    authenticator = "EXTERNALBROWSER"

    engine = create_engine(
        URL(
            account=account,
            region=region,
            user=username,
            authenticator=authenticator,
            database=database,
            warehouse=warehouse,
        )
    )
    try:
        connection = engine.connect()
        df = pd.read_sql(query, connection)
    finally:
        connection.close()
        engine.dispose

    return df


def bootstrap_generator(
    data: pd.DataFrame, n_bootstrap: int, sample_size: float = False
) -> Generator[
    pd.DataFrame, None, None
]:  # TODO: typing of sample_size and verify return type
    """Generates bootstrap samples of data.

    Args:
        data (pd.DataFrame): data
        n_bootstrap (int): number of bootstrap samples
        sample_size (bool, optional): size of each bootstrap sample. Defaults to False.

    Yields:
        pd.DataFrame: Bootstrapped sample
    """

    if not sample_size:
        sample_size = len(data)

    for _ in range(n_bootstrap):
        yield data.sample(n=sample_size, replace=True)


# NOTE: Say the sample split is 0.3, i think that for some data the control will be 0.3 and for some it will be the treatment, depending on which will end up first when unique() is taken
def subsample_generator(
    data: pd.DataFrame,
    sample_size: int,
    n_bootstrap: int,
    treatment_col: str,
    sample_split=False,
) -> Generator[
    pd.DataFrame, None, None
]:  # TODO: typing of sample_split and verify generator
    """Generates subsamples of experiment data.

    Args:
        data (pd.DataFrame): experiment data
        sample_size (int): sample size of each subsample
        n_bootstrap (int): number of subsamples
        treatment_col (str): name of treatment column
        sample_split (bool, optional): custom sample split proportion. Defaults to False.

    Yields:
        pd.DataFrame: A subsample of experiment data.
    """

    unique_treatments = data[treatment_col].unique()

    assert len(unique_treatments) == 2

    # mask = data[treatment_col]==unique_treatments[0]
    # data_0 = data.loc[mask]
    # data_1 = data.loc[-mask]

    data_0 = data.loc[data[treatment_col] == unique_treatments[0]]
    data_1 = data.loc[data[treatment_col] == unique_treatments[1]]

    if not sample_split:
        sample_split = len(data_0) / len(data)
    n0 = int(sample_size * sample_split)
    n1 = sample_size - n0

    for bootstrapped_0, bootstrapped_1 in zip(
        bootstrap_generator(data_0, n_bootstrap, n0),
        bootstrap_generator(data_1, n_bootstrap, n1),
    ):
        yield pd.concat([bootstrapped_0, bootstrapped_1]).sample(frac=1).reset_index(
            drop=True
        )


def split_dataframe(df: pd.DataFrame, K: int) -> List[np.array]:
    """Splits a dataframe into K splits uniformly at random

    Args:
        df (pd.DataFrame): data
        K (int): number of splits

    Returns:
        List[np.array]: list of indices of splits
    """

    # Shuffle the indices of the DataFrame
    shuffled_indices = np.random.permutation(df.index)

    # Calculate the approximate size of each split
    split_size = len(df) // K

    # Initialize an empty list to store the split indices
    split_indices_list = []
    index_to_split_map = {}

    # Split the DataFrame into K approximately equal-sized parts
    for i in range(K):
        start = i * split_size
        end = (i + 1) * split_size if i < K - 1 else len(df)

        # Extract the current split indices based on shuffled indices
        current_split_indices = shuffled_indices[start:end]

        # Append the split indices to the list
        split_indices_list.append(current_split_indices)

        for ind in current_split_indices:
            index_to_split_map[ind] = i

    return split_indices_list, index_to_split_map


def bootstrap_data(
    data: pd.DataFrame, n_bootstrap: int, sample_size: float = False
) -> np.array:
    """Returns the indices of bootstrap samples of data."""
    N = len(data)

    if not sample_size:
        sample_size = N

    # assert sample_size <= N

    # bootstrap_indices = np.zeros((n_bootstrap, sample_size), dtype=int)

    # for i in range(n_bootstrap):

    bootstrap_indices = np.random.choice(
        data.index, size=(n_bootstrap, sample_size), replace=True
    )

    return bootstrap_indices


def subsample_data(
    data: pd.DataFrame,
    n_bootstrap: int,
    treatment_column: str,
    sample_size: int = False,
    sample_split=False,
) -> np.array:
    """Returns the indices of subsamples of experiment data."""

    N = len(data)

    if not sample_size:
        sample_size = N

    assert set(data[treatment_column].unique()) == {0, 1}

    # mask = data[treatment_col]==unique_treatments[0]
    # data_0 = data.loc[mask]
    # data_1 = data.loc[-mask]

    data_0 = data.loc[data[treatment_column] == 0]
    data_1 = data.loc[data[treatment_column] == 1]

    if not sample_split:
        sample_split = len(data_0) / len(data)
    n0 = int(sample_size * sample_split)
    n1 = sample_size - n0

    bootstrap_indices_0 = bootstrap_data(data_0, n_bootstrap, n0)
    bootstrap_indices_1 = bootstrap_data(data_1, n_bootstrap, n1)

    return np.concatenate([bootstrap_indices_0, bootstrap_indices_1], axis=1)


def aaify(data: pd.DataFrame, treatment_column: str, frac_control=False):
    """Turn an A/B experiment data into A/A experiment data by randomly assigning treatment to control.

    Args:
        data (pd.DataFrame): experiment data
        treatment_column (str): name of column containing treatment flags
        frac_control (bool, optional): fraction of data to be assigned to control. Defaults to False. # TODO: typing

    Returns:
        pd.DataFrame: experiment data with treatment filtered out and new treatment assignment
    """

    aa_df = data.loc[data[treatment_column] == 0]

    N = len(aa_df)

    p = frac_control if frac_control else 0.5

    treatment_assignment = np.random.binomial(1, p, size=N)

    aa_df[treatment_column] = treatment_assignment

    return aa_df.reset_index(drop=True)


def add_synthetic_effect(
    data: pd.DataFrame, treatment_column: str, target_column: str, effect_size: float
):
    """Add a synthetic effect to the experiment data.

    Args:
        data (pd.DataFrame): experiment data
        treatment_column (str): name of column containing treatment flags
        effect_size (float): effect size
        treatment_column (str): name of column containing effect

    Returns:
        pd.DataFrame: experiment data with synthetic effect
    """

    N = len(data)

    synthetic_effect_data = data.copy()

    synthetic_effect_data[target_column] = (
        synthetic_effect_data[target_column]
        + synthetic_effect_data[treatment_column] * effect_size
    )  # + np.random.normal(0, 1, size=N)

    return synthetic_effect_data


def diff_in_means(data: pd.DataFrame, treatment_column: str, target_column: str):
    """Calculate the difference in means between treatment and control groups.

    Args:
        data (pd.DataFrame): experiment data
        treatment_column (str): name of column containing treatment flags
        target_column (str): name of column containing target variable

    Returns:
        float: difference in means
    """

    treatment_mean = data.loc[data[treatment_column] == 1, target_column].mean()
    control_mean = data.loc[data[treatment_column] == 0, target_column].mean()

    return treatment_mean - control_mean
