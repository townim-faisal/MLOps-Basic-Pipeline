# Pipeline Setup

Run following commands from root directory.

For more, please read [here](https://airflow.apache.org/docs/apache-airflow/stable/start/local.html).

```bash
export AIRFLOW_HOME=${PWD}/pipelines
airflow db init
```
Change following in `airflow.cfg` file.
```
load_examples = False
```

Then run these commands.
```bash
airflow db reset
airflow config get-value core sql_alchemy_conn
airflow config get-value core executor
airflow users create --username admin --firstname admin --lastname admin --role Admin --email admin@admin.org
export AIRFLOW_HOME=${PWD}/pipelines
airflow dags list
airflow webserver --port 8080
```

In another terminal, run this to monitor.
```bash
export AIRFLOW_HOME=${PWD}/pipelines
airflow scheduler
```

# Further Reading

- [Core concepts](https://airflow.apache.org/docs/apache-airflow/1.10.6/concepts.html)
- [Production deployment](https://airflow.apache.org/docs/apache-airflow/stable/production-deployment.html)
- [Set up database](https://airflow.apache.org/docs/apache-airflow/2.2.3/howto/set-up-database.html)
- [Schedule interval](https://airflow.apache.org/docs/apache-airflow/stable/dag-run.html#dag-runs)
- [Catchup](https://airflow.apache.org/docs/apache-airflow/stable/dag-run.html#catchup) : [blog](https://medium.com/nerd-for-tech/airflow-catchup-backfill-demystified-355def1b6f92)
- [DAG](https://airflow.apache.org/docs/apache-airflow/stable/concepts/dags.htm)