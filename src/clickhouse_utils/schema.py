from typing import Union, Sequence

import pandas as pd
import numpy as np
from clickhouse_driver import Client
from sqlalchemy import create_engine
from functools import lru_cache

type_dict = {
    np.dtype("int64"): "Int64",
    np.dtype("int32"): "Int32",
    np.dtype("O"): "String",
    np.dtype("float64"): "Float64",
    np.dtype("float32"): "Float32",
    np.dtype("<M8[ns]"): "DateTime",
    np.dtype("bool"): "UInt8",
}


@lru_cache(1024)  # caches results for speed
def dtypes_from_table(host: str, database: str, table: str = None, query: str = None):

    ch_url = f"clickhouse+native://{host}/{database}"
    ch_engine = create_engine(ch_url, echo=False)

    if query is None:
        query = f"select *  from {database}.{table} limit 1"
    else:
        query += " limit 1"

    df = pd.read_sql(query, ch_engine)
    return df.dtypes


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
