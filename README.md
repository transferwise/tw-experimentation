# clickhouse_utils
A collection of Clickhouse-related utilities

To spin up a docker-compose cluster with Kafka and Clickhouse, run `docker-compose up -d` in this directory. 
Warning: the ports it exposes won't be visible outside of the machine it runs on unless you port map with ssh or similar.

To bring up a standalone Clickhouse instance visible outside of this machine, run 
`docker run -d --network=host --name clickhouse-server --ulimit nofile=262144:262144 -p 8123:8123 -p 9000:9000 --volume=storage:/var/lib/clickhouse yandex/clickhouse-server`