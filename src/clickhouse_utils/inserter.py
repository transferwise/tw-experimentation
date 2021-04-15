import datetime
import glob
from typing import Iterable, Callable

import numpy as np
import pandas as pd
from pandahouse import to_clickhouse


def clickhouse_insert_dataframes(
    data: Iterable[pd.DataFrame],
    database_name: str,
    table_name: str,
    host: str = "localhost",
    http_port: str = 8123,
) -> None:
    connection = {"host": f"http://{host}:{http_port}", "database": database_name}
    for this_df in data:
        print(
            len(this_df),
            datetime.datetime.now(),
        )

        # CH is particular about time formatting
        for col, t in this_df.dtypes.iteritems():
            if t == np.dtype("<M8[ns]"):
                this_df[col] = this_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        to_clickhouse(this_df, table=table_name, connection=connection, index=False)


def jsons_files_to_iterable(pattern: str):
    return files_to_df_iterable(pattern, lambda x: pd.read_json(x, lines=True))


def csv_files_to_iterable(pattern: str):
    return files_to_df_iterable(pattern, lambda x: pd.read_csv(x))


def files_to_df_iterable(pattern: str, loader: Callable):
    files = glob.glob(pattern)
    for file in files:
        print(file)
        yield (loader(file))
