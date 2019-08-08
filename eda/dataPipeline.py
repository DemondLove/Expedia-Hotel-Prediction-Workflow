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

import src.dataPreparation as dp

parentPath = '/'.join(sys.path[1].split('/')[:-1])
parentPathOne = '/'.join(sys.path[1].split('/')[:-1])
print("parentPathOne", parentPathOne)

logging.basicConfig(filename=parentPath+'/logging/dataPipelineLogging.log',level=logging.DEBUG)

########

tic = time.process_time()

df = pd.read_csv(parentPath+'/data/pd___dfExpediaSample.csv')

toc = time.process_time()

logging.info(str(datetime.datetime.now()) + ': ' + "Read CSV Time elapsed: "+ str(round(toc-tic, 3))+ " seconds")

########

tic = time.process_time()

df = dp.updateIDFieldsToCategoricalFeatures(df)

toc = time.process_time()

logging.info(str(datetime.datetime.now()) + ': ' + "updateIDFieldsToCategoricalFeatures Time elapsed: "+ str(round(toc-tic, 3))+ " seconds")

########

tic = time.process_time()

df = dp.updateISFieldsToBooleanFeatures(df)

toc = time.process_time()

logging.info(str(datetime.datetime.now()) + ': ' + "updateISFieldsToBooleanFeatures Time elapsed: "+ str(round(toc-tic, 3))+ " seconds")

########

tic = time.process_time()

df = dp.removeHighCardinalityFeatures(df)

toc = time.process_time()

logging.info(str(datetime.datetime.now()) + ': ' + "removeHighCardinalityFeatures Time elapsed: "+ str(round(toc-tic, 3))+ " seconds")

########

tic = time.process_time()

df = dp.removeHighNULLCntFeatures(df)

toc = time.process_time()

logging.info(str(datetime.datetime.now()) + ': ' + "removeHighNULLCntFeatures Time elapsed: "+ str(round(toc-tic, 3))+ " seconds")

########

tic = time.process_time()

df = dp.removeRemainingRecordsWithNULLS(df)

toc = time.process_time()

logging.info(str(datetime.datetime.now()) + ': ' + "removeRemainingRecordsWithNULLS Time elapsed: "+ str(round(toc-tic, 3))+ " seconds")

########

tic = time.process_time()

df = dp.convertCategoricalVariablesToDummyVariables(df)

toc = time.process_time()

logging.info(str(datetime.datetime.now()) + ': ' + "convertCategoricalVariablesToDummyVariables Time elapsed: "+ str(round(toc-tic, 3))+ " seconds")

########

tic = time.process_time()

#pd_dfExpediaSample.to_csv(parentPath+'/data/pd_FullCleansedDataset.csv', index=False)

toc = time.process_time()

logging.info(str(datetime.datetime.now()) + ': ' + "Export CSV Time elapsed: "+ str(round(toc-tic, 3))+ " seconds")

########