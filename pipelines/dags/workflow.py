from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator

from datetime import datetime
from random import randint

def choosing_best_model(ti):
    accuracies = ti.xcom_pull(task_ids=[
        'training_model_A',
        'training_model_B',
        'training_model_C'
        ])
    status = ''
    if max(accuracies) > 8:
        status = 'accurate'
    else: 
        status = 'inaccurate'
    print(status)
    return status

def training_model(model):
    a =  randint(1, 10)
    print(model, ':', a)
    return a

default_args = {
    'owner': 'airflow',
    'depends_on_past': False
}

with DAG("test",
    start_date=datetime(2021, 1 ,1), 
    schedule_interval='@daily', 
    default_args=default_args,
    catchup=False
) as dag:

    training_model_tasks = [
        PythonOperator(
            task_id=f"training_model_{model_id}",
            python_callable=training_model,
            op_kwargs={
                "model": model_id
            }
        ) for model_id in ['A', 'B', 'C']
    ]

    choosing_best_model_task = BranchPythonOperator(
        task_id="choosing_best_model",
        python_callable=choosing_best_model
    )
    
    accurate = BashOperator(
        task_id="accurate",
        bash_command="echo 'accurate'"
    )

    inaccurate = BashOperator(
        task_id="inaccurate",
        bash_command=" echo 'inaccurate'"
    )

    training_model_tasks >> choosing_best_model_task >> [accurate, inaccurate]