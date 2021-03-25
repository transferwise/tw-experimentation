from typing import Iterable
import glob
import datetime

import pandas as pd
import numpy as np

from clickhouse_driver import Client
from pandahouse import to_clickhouse

from clickhouse_utils.schema import initialize_schema


def clickhouse_insert(
    data: Iterable,
    database_name: str,
    table_name: str,
    host: str = "localhost",
    http_port: str = 8123,
) -> None:
    connection = {"host": f"http://{host}:{http_port}", "database": database_name}
    for this_df in data:
        print(
            len(this_df), datetime.datetime.now(),
        )
        # CH is particular about time formatting
        for col, t in dtypes.iteritems():
            if t == np.dtype("<M8[ns]"):
                this_df[col] = this_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        to_clickhouse(this_df, table=table_name, connection=connection, index=False)


def jsons_files_to_iterable(pattern: str):
    files = glob.glob(pattern)
    for file in files:
        print(file)
        yield (pd.read_json(file, lines=True))


def json_header_to_iterable(pattern: str):
    files = glob.glob(pattern)
    for file in files:
        print(file)
        yield (pd.read_json(file, lines=True))


if __name__ == "__main__":
    import os

    root_dir = os.path.realpath("../../..")

    # this script needs at least one json file to exist, to infer the schema from
    data_loc = os.path.join(
        root_dir, "treasury_accounting/druid/raw_data/ledger_2021-02-01_0.json"
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

    host = "172.31.50.194"
    port = 9000
    http_port = 8123

    client = Client(host=host, port=port)
    dtypes = next(iter(data)).dtypes

    initialize_schema(
        client,
        database_name="ledger",
        table_name="ledger",
        dtypes=dtypes,
        uid_name="ID",
        time_col="TIMESTAMP",
        high_granularity=high_cardinality,
        flush_table=True,
    )

    if (
        False
    ):  # this method is way too slow, inject directly from command line with clickhouse-client instead
        dtypes = clickhouse_insert(
            data, "ledger", "ledger", host=host, http_port=http_port,
        )
