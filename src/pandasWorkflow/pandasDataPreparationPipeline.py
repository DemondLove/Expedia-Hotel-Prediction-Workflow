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

import src.pandasWorkflow.pandasDataPreparationUtils as dp

parentPath = '/'.join(sys.path[1].split('/')[:-2])

logging.basicConfig(filename=parentPath+'/logging/dataPipelineLogging.log',level=logging.DEBUG)



dp.logStartOfDataPipeline()

########

df = dp.importDataset(parentPath=parentPath, inputFilePath='/data/pd_dfExpediaSample.csv')

########

df = dp.updateIDFieldsToCategoricalFeatures(df)

########

df = dp.updateISFieldsToBooleanFeatures(df)

########

df = dp.removeHighCardinalityFeatures(df)

########

df = dp.removeHighNULLCntFeatures(df)

########

df = dp.removeRemainingRecordsWithNULLS(df)

########

df = dp.convertCategoricalVariablesToDummyVariables(df)

########

dp.exportDataset(df=df, outputPath='/data/pd_CleansedDataset.csv')

########

dp.logEndOfDataPipeline()
