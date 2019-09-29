import sys
import os
import inspect
import numpy as np
import pandas as pd
import unittest

sys.path.insert(0, '/'.join(sys.path[0].split('/')[:-2]))

import src.pandasWorkflow.pandasDataPreparationUtils as dp

from importlib import reload

reload(dp)

class TestdataPreparation(unittest.TestCase):
    
    # Ensure that updateIDFieldsToCategoricalFeatures is properly updating the ID fields to be categorical features
    def test_updateIDFieldsToCategoricalFeatures(self):
        parentPath = '/'.join(sys.path[0].split('/')[:-1])
        df = pd.read_csv(parentPath+'/Expedia-Hotel-Prediction-Workflow/data/pd_dfExpediaSample.csv')
        
        df = dp.updateIDFieldsToCategoricalFeatures(df)
        
        self.assertEqual(str(df['site_name'].dtype), 'category')
    
    # Ensure that updateISFieldsToBooleanFeatures is properly updating the boolean fields to the correct datatype
    def test_updateISFieldsToBooleanFeatures(self):
        parentPath = '/'.join(sys.path[0].split('/')[:-1])
        df = pd.read_csv(parentPath+'/Expedia-Hotel-Prediction-Workflow/data/pd_dfExpediaSample.csv')
        
        df = dp.updateISFieldsToBooleanFeatures(df)
        
        self.assertEqual(str(df['is_mobile'].dtype), 'bool')
        
    # Ensure that removeHighCardinalityFeatures is properly removing the high cardinality variables
    def test_removeHighCardinalityFeatures(self):
        parentPath = '/'.join(sys.path[0].split('/')[:-1])
        df = pd.read_csv(parentPath+'/Expedia-Hotel-Prediction-Workflow/data/pd_dfExpediaSample.csv')
        
        df = dp.removeHighCardinalityFeatures(df)
        
        if 'date_time' in df.columns:
            A = 1
        else:
            A = 0
        
        self.assertEqual(A, 0)
    
    # Ensure that removeHighNULLCntFeatures is properly removing variables with abnormally high number of missing values
    def test_removeHighNULLCntFeatures(self):
        parentPath = '/'.join(sys.path[0].split('/')[:-1])
        df = pd.read_csv(parentPath+'/Expedia-Hotel-Prediction-Workflow/data/pd_dfExpediaSample.csv')
        
        df = dp.removeHighNULLCntFeatures(df)
        
        if 'orig_destination_distance' in df.columns:
            A = 1
        else:
            A = 0
        
        self.assertEqual(A, 0)

    # Ensure that convertCategoricalVariablesToDummyVariables is properly converting categorical variables into dummy variables
    def test_convertCategoricalVariablesToDummyVariables(self):
        parentPath = '/'.join(sys.path[0].split('/')[:-1])
        df = pd.read_csv(parentPath+'/Expedia-Hotel-Prediction-Workflow/data/pd_dfExpediaSample.csv')
        
        df = dp.updateIDFieldsToCategoricalFeatures(df)
        
        df = dp.convertCategoricalVariablesToDummyVariables(df[['posa_continent', 'hotel_continent']])
        if 'posa_continent' in df.columns:
            A = 1
        else:
            A = 0
        
        self.assertEqual(A, 0)
        
if __name__ == '__main__':
    unittest.main()
