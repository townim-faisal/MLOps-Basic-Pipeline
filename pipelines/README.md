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
executor = LocalExecutor
dags_folder = <absolute path>/pipelines/dags
sql_alchemy_conn = sqlite:////<absolute path>/pipelines/airflow.db
```

Then run these commands.
```bash
airflow db reset
airflow config get-value core sql_alchemy_conn
airflow config get-value core executor
airflow users create --username admin --firstname admin --lastname admin --role Admin --email admin@admin.org
```

If you have already setup the project, do not need to initialize db. Run this only.
```bash
export AIRFLOW_HOME=${PWD}/pipelines
airflow dags list
airflow webserver --port 8080
```

In another terminal, run this to monitor.
```bash
export AIRFLOW_HOME=${PWD}/pipelines
airflow scheduler
```

For production, try to follow [this](https://airflow.apache.org/docs/apache-airflow/stable/production-deployment.html). Also, follow [this](https://airflow.apache.org/docs/apache-airflow/2.2.3/howto/set-up-database.html) to setup other database.