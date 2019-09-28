import datetime as dt
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

def func():
	print('Airflow Dag is alive and well!')

def func2():
	print('On to the Next One!')

default_args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2018, 9, 24, 10, 00, 00),
    'concurrency': 1,
    'retries': 0
}

divvy_dag = DAG(
	'divvy_dirty_money',
	description='My First locally hosted Airflow DAG',
    start_date= dt.datetime(2018, 9, 24, 10, 00, 00),
	schedule_interval='@daily'
	)

task = PythonOperator(
	task_id='hello_world',
	python_callable=func,
	dag=divvy_dag)

task2 = PythonOperator(
	task_id='hello_world_again',
	python_callable=func2,
	dag=divvy_dag)

task >> task2