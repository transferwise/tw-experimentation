from typing import Callable, Sequence, Dict, Iterable
import glob

import pandas as pd
import numpy as np

from clickhouse_driver import Client
from pandahouse import to_clickhouse

from clickhouse_utils.schema import make_clickhouse_schema


def clickhouse_insert(
    data: Iterable,
    database_name: str,
    table_name: str,
    uid_name: str,
    time_col: str,
    high_granularity: Sequence = (),
    host: str = "localhost",
    port: str = 9000,
    http_port: str = 8123,
    flush_table: bool = False,
) -> None:
    client = Client(host=host, port=port)
    connection = {"host": f"http://{host}:{http_port}", "database": database_name}
    dtypes = None
    for this_df in data:
        print(len(this_df))
        initialized = False
        if not initialized:
            client.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")

            if flush_table:
                client.execute(f"DROP TABLE IF EXISTS {database_name}.{table_name}")
            dtypes = this_df.dtypes
            schema = make_clickhouse_schema(
                dtypes,
                f"{database_name}.{table_name}",
                (uid_name, time_col),
                high_granularity=high_granularity,
            )
            print(schema)
            client.execute(schema)
            initialized = True

        # CH is particular about time formatting
        for col, t in dtypes.iteritems():
            if t == np.dtype("<M8[ns]"):
                print(col)
                this_df[col] = this_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        to_clickhouse(this_df, table=table_name, connection=connection, index=False)
    return dtypes


def jsons_files_to_iterable(pattern: str):
    files = glob.glob(pattern)
    for file in files:
        yield (pd.read_json(file, lines=True))


if __name__ == "__main__":
    import os

    root_dir = os.path.realpath("../../..")
    data_loc = os.path.join(
        root_dir, "treasury_accounting/druid/raw_data/ledger_*.json"
    )
    data = jsons_files_to_iterable(data_loc)

    high_cardinality = [
        "CORRELATIONID",
        "FLOWID",
        "ID",
        "LINKEDFLOWID",
        "REVERSEDTRANSACTIONID",
        "EVENT_TIMESTAMP",
    ]

    dtypes = clickhouse_insert(
        data,
        "ledger",
        "ledger",
        "ID",
        "TIMESTAMP",
        high_cardinality,
        host="localhost",
        port=9001,
        http_port=8124,
    )
