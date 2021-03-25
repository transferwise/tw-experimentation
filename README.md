# clickhouse_utils
A collection of Clickhouse-related utilities

To spin up a docker-compose cluster with Kafka and Clickhouse, run `docker-compose up -d` in this directory. 
Warning: the ports it exposes won't be visible outside of the machine it runs on unless you port map with ssh or similar.

For an example of data ingestion from Kafka, see `src/notebooks/kafka_import.ipynb`

To bring up a standalone Clickhouse instance visible outside of the machine it runs on, run 
`docker run -d --network=host --name clickhouse-server --ulimit nofile=262144:262144 --volume=storage_standalone:/var/lib/clickhouse yandex/clickhouse-server`

**This will only work on Linux, not Mac or Windows!**

## Ingesting the fi
To bulk-ingest json files produced by the json dumper (meant for Druid ingestion), run
` ls ./ledger_2021-02-01* | while read p; do cat $p | clickhouse-client --database=ledger --query="INSERT INTO ledger FORMAT JSONEachRow"; done`