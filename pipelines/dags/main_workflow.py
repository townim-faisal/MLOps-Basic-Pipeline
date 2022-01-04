from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from datetime import datetime
from pathlib import Path
import os, sys

from data.eda import eda
from data.preprocess import preprocess

root_dir = Path(__file__).parent.parent.absolute()
print(root_dir)

default_args = {
    'owner': 'airflow',
    'depends_on_past': True 
}

def eda_task(file_name="params/param.yaml"):
    status = eda(file_name)
    return status

def preprocess_task(**kwargs):
    ti = kwargs['ti']
    file_name = kwargs['file_name']
    status = ti.xcom_pull(task_ids='eda')
    if status:
        preprocess(file_name)
        print('Completed')
    else:
        print('Failed to start because eda has not been completed')


# Data Pipeline
with DAG("data_pipeline",
    start_date=datetime(2021, 1 ,1), 
    schedule_interval=None, 
    default_args=default_args,
    catchup=False
) as data_dag:

    eda_task_id = PythonOperator(
        task_id='eda',
        python_callable=eda_task,
        op_kwargs={
            "file_name" : os.path.join(root_dir, "dags", "data", "params/param.yaml")
        },
        dag=data_dag
    )

    preprocess_task_id = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess_task,
        op_kwargs={
            "file_name" : os.path.join(root_dir, "dags", "data", "params/param.yaml")
        },
        dag=data_dag
    )

    eda_task_id >> preprocess_task_id

# Model Pipeline
with DAG("model_pipeline",
    start_date=datetime(2021, 1 ,1), 
    schedule_interval=None, 
    default_args=default_args,
    catchup=False
) as model_dag:

    model_dir = os.path.join(root_dir, 'dags', 'model')
    model_task_id = BashOperator(
        task_id="model_train",
        bash_command=f"cd {model_dir} && python main.py",
        dag=model_dag
    )

    model_task_id






