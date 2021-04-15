from typing import List, Union, Sequence

import pandas as pd
import numpy as np
from clickhouse_driver import Client

type_dict = {
    np.dtype("int64"): "Int64",
    np.dtype("int32"): "Int32",
    np.dtype("O"): "String",
    np.dtype("float64"): "Float64",
    np.dtype("float32"): "Float32",
    np.dtype("<M8[ns]"): "DateTime",
    np.dtype("bool"): "UInt8",
}


def normalize_types(x: np.dtype):
    if x == np.dtype("float64"):
        return np.float32
    else:
        return x


def make_clickhouse_schema(
    dtypes: pd.Series,
    table_name: str,
    order_by: Union[str, Sequence[str]],
    high_granularity: Sequence = (),
):

    cols = []
    for cname, ctype in dtypes.iteritems():
        cht = type_dict[ctype]
        if cht == "String" and cname not in high_granularity:
            cht = "LowCardinality(String)"
        cols.append(f"    `{cname}` {cht},")

    all_cols = "\n".join(cols)[:-1]  # drop the last comma

    if len(order_by) == 1:
        order_by = order_by[0]

    if isinstance(order_by, str):
        order_str = f"`{order_by}`"
    else:
        inner = ", ".join([f"`{c}`" for c in order_by])
        order_str = f"({inner})"

    schema = f"""
                CREATE TABLE IF NOT EXISTS {table_name}
                (
                {all_cols}
                )
                ENGINE = MergeTree()
                ORDER BY {order_str}
            """
    return schema


def initialize_schema(
    client: Client,
    database_name: str,
    table_name: str,
    dtypes: pd.Series,
    order_by: Sequence[str],
    high_cardinality: Sequence = (),
    flush_table: bool = False,
):
    client.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")

    if flush_table:
        client.execute(f"DROP TABLE IF EXISTS {database_name}.{table_name}")

    schema = make_clickhouse_schema(
        dtypes,
        f"{database_name}.{table_name}",
        order_by=order_by,
        high_granularity=high_cardinality,
    )
    print(schema)
    client.execute(schema)
