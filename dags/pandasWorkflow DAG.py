import datetime as dt
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import os
import sys
import time
import logging
import inspect
import datetime
import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.stats import kurtosis

sys.path.insert(0, '/'.join(sys.path[0].split('/')[:-1]))

import src.pandasWorkflow.pandasDataPreparationUtils as dp

parentPath = '/'.join(sys.path[1].split('/')[:-1])

logging.basicConfig(filename=parentPath+'/logging/dataPipelineLogging.log',level=logging.DEBUG)

dp.logStartOfDataPipeline()
########
df = dp.importDataset(parentPath=parentPath, inputFilePath='/data/pd_dfExpediaSample.csv')
########
df = dp.updateIDFieldsToCategoricalFeatures
########
df = dp.updateISFieldsToBooleanFeatures
########
df = dp.removeHighCardinalityFeatures
########
df = dp.removeHighNULLCntFeatures
########
df = dp.removeRemainingRecordsWithNULLS
########
df = dp.convertCategoricalVariablesToDummyVariables
########
dp.exportDataset(df=df, outputPath='/data/pd_CleansedDataset.csv')
########
dp.logEndOfDataPipeline()

# def func():
# 	print('Airflow Dag is alive and well!')

# def func2():
# 	print('On to the Next One!')

default_args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2018, 9, 24, 10, 00, 00),
    'concurrency': 1,
    'retries': 0
}

pandasWorkflow_dag = DAG(
	'pandasWorkflow_dag',
	description='Airflow DAG to process the pandasWorkflow DAG',
    start_date= dt.datetime(2018, 9, 24, 10, 00, 00),
	schedule_interval='@daily'
	)

logStartOfDataPipeline = PythonOperator(
	task_id='logStartOfDataPipeline',
	python_callable=dp.logStartOfDataPipeline,
	dag=pandasWorkflow_dag)

importDataset = PythonOperator(
	task_id='importDataset',
	python_callable=dp.importDataset,
	dag=pandasWorkflow_dag)

updateIDFieldsToCategoricalFeatures = PythonOperator(
	task_id='updateIDFieldsToCategoricalFeatures',
	python_callable=dp.updateIDFieldsToCategoricalFeatures,
	dag=pandasWorkflow_dag)

updateISFieldsToBooleanFeatures = PythonOperator(
	task_id='updateISFieldsToBooleanFeatures',
	python_callable=dp.updateISFieldsToBooleanFeatures,
	dag=pandasWorkflow_dag)

removeHighCardinalityFeatures = PythonOperator(
	task_id='removeHighCardinalityFeatures',
	python_callable=dp.removeHighCardinalityFeatures,
	dag=pandasWorkflow_dag)

removeHighNULLCntFeatures = PythonOperator(
	task_id='removeHighNULLCntFeatures',
	python_callable=dp.removeHighNULLCntFeatures,
	dag=pandasWorkflow_dag)

removeRemainingRecordsWithNULLS = PythonOperator(
	task_id='removeRemainingRecordsWithNULLS',
	python_callable=dp.removeRemainingRecordsWithNULLS,
	dag=pandasWorkflow_dag)

convertCategoricalVariablesToDummyVariables = PythonOperator(
	task_id='convertCategoricalVariablesToDummyVariables',
	python_callable=dp.convertCategoricalVariablesToDummyVariables,
	dag=pandasWorkflow_dag)

exportDataset = PythonOperator(
	task_id='exportDataset',
	python_callable=dp.exportDataset,
	dag=pandasWorkflow_dag)

logEndOfDataPipeline = PythonOperator(
	task_id='logEndOfDataPipeline',
	python_callable=dp.logEndOfDataPipeline,
	dag=pandasWorkflow_dag)

logStartOfDataPipeline >> importDataset >> updateIDFieldsToCategoricalFeatures >> updateISFieldsToBooleanFeatures >> removeHighCardinalityFeatures >> removeHighNULLCntFeatures >> removeRemainingRecordsWithNULLS >> convertCategoricalVariablesToDummyVariables >> exportDataset >> logEndOfDataPipeline 