from clickhouse_driver import Client

from clickhouse_utils.inserter import jsons_files_to_iterable
from clickhouse_utils.schema import initialize_schema

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
        high_cardinality=high_cardinality,
        flush_table=True,
    )

    if (
        False
    ):  # this method is way too slow, inject directly from command line with clickhouse-client instead
        dtypes = clickhouse_insert(
            data,
            "ledger",
            "ledger",
            host=host,
            http_port=http_port,
        )
