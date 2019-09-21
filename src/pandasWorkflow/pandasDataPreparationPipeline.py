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

sys.path.insert(0, '/'.join(sys.path[0].split('/')[:-2]))

import src.pandasWorkflow.pandasDataPreparation as dp

parentPath = '/'.join(sys.path[1].split('/')[:-2])

logging.basicConfig(filename=parentPath+'/logging/dataPipelineLogging.log',level=logging.DEBUG)

print(parentPath)

########

tic = time.perf_counter()

df = pd.read_csv(parentPath+'/data/pd___dfExpediaSample.csv')

toc = time.perf_counter()

logging.info(str(datetime.datetime.now()) + ': ' + "Import CSV Time elapsed: "+ str(round(toc-tic, 3))+ " seconds")

########

tic = time.perf_counter()

df = dp.updateIDFieldsToCategoricalFeatures(df)

toc = time.perf_counter()

logging.info(str(datetime.datetime.now()) + ': ' + "updateIDFieldsToCategoricalFeatures Time elapsed: "+ str(round(toc-tic, 3))+ " seconds")

########

tic = time.perf_counter()

df = dp.updateISFieldsToBooleanFeatures(df)

toc = time.perf_counter()

logging.info(str(datetime.datetime.now()) + ': ' + "updateISFieldsToBooleanFeatures Time elapsed: "+ str(round(toc-tic, 3))+ " seconds")

########

tic = time.perf_counter()

df = dp.removeHighCardinalityFeatures(df)

toc = time.perf_counter()

logging.info(str(datetime.datetime.now()) + ': ' + "removeHighCardinalityFeatures Time elapsed: "+ str(round(toc-tic, 3))+ " seconds")

########

tic = time.perf_counter()

df = dp.removeHighNULLCntFeatures(df)

toc = time.perf_counter()

logging.info(str(datetime.datetime.now()) + ': ' + "removeHighNULLCntFeatures Time elapsed: "+ str(round(toc-tic, 3))+ " seconds")

########

tic = time.perf_counter()

df = dp.removeRemainingRecordsWithNULLS(df)

toc = time.perf_counter()

logging.info(str(datetime.datetime.now()) + ': ' + "removeRemainingRecordsWithNULLS Time elapsed: "+ str(round(toc-tic, 3))+ " seconds")

########

tic = time.perf_counter()

df = dp.convertCategoricalVariablesToDummyVariables(df)

toc = time.perf_counter()

logging.info(str(datetime.datetime.now()) + ': ' + "convertCategoricalVariablesToDummyVariables Time elapsed: "+ str(round(toc-tic, 3))+ " seconds")

########

tic = time.perf_counter()

df.to_csv(parentPath+'/data/pd_CleansedDataset.csv', index=False)

toc = time.perf_counter()

logging.info(str(datetime.datetime.now()) + ': ' + "Export CSV Time elapsed: "+ str(round(toc-tic, 3))+ " seconds")

########

print('dataPreparationPipeline Complete!')