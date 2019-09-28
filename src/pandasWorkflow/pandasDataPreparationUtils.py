import sys
import os
import inspect
import numpy as np
import pandas as pd
import time
import os,sys,inspect
import logging
import datetime

parentPath = '/'.join(sys.path[1].split('/')[:-2])

logging.basicConfig(filename=parentPath+'/logs/dataPipelineLogging.log',level=logging.DEBUG)

def updateIDFieldsToCategoricalFeatures(df):
    '''
    Update the ID fields to be categorical features
    Specifically, site_name, posa_continent, user_location_country, user_location_region, user_location_city, user_id, channel, srch_destination_id, srch_destination_type_id, hotel_continent, hotel_country, hotel_market
    
    Parameters:
    df (pd.DataFrame): Input dataset from Expedia Hotel Recommendations
    
    Returns:
    df (pd.DataFrame): Converted dataFrame with the ID fields containing updated datatypes
    '''
    try:
        tic = time.perf_counter()
        
        df['site_name'] = df['site_name'].astype('category')
        df['posa_continent'] = df['posa_continent'].astype('category')
        df['user_location_country'] = df['user_location_country'].astype('category')
        df['channel'] = df['channel'].astype('category')
        df['srch_destination_type_id'] = df['srch_destination_type_id'].astype('category')
        df['hotel_continent'] = df['hotel_continent'].astype('category')
        df['hotel_country'] = df['hotel_country'].astype('category')
        
        toc = time.perf_counter()

        logging.info(str(datetime.datetime.now()) + ': ' + 'updateIDFieldsToCategoricalFeatures Time elapsed: '+ str(round(toc-tic, 3))+ ' seconds')

        return df
    except Exception as e:
        logging.info(str(datetime.datetime.now()) + ': ' + 'An Error has Occured in updateIDFieldsToCategoricalFeatures: ' + str(e))

def updateISFieldsToBooleanFeatures(df):
    '''
    Update the boolean fields to the correct datatype
    Specifically, is_mobile, is_package, is_booking
    
    Parameters:
    df (pd.DataFrame): Input dataset from Expedia Hotel Recommendations
    
    Returns:
    df (pd.DataFrame): Converted dataFrame with the boolean fields updated to the correct datatype
    '''
    
    try:
        tic = time.perf_counter()
        
        df['is_mobile'] = df['is_mobile'].astype('bool')
        df['is_package'] = df['is_package'].astype('bool')
        df['is_booking'] = df['is_booking'].astype('bool')
        
        toc = time.perf_counter()

        logging.info(str(datetime.datetime.now()) + ': ' + 'updateISFieldsToBooleanFeatures Time elapsed: '+ str(round(toc-tic, 3))+ ' seconds')

        return df
    except Exception as e:
        logging.info(str(datetime.datetime.now()) + ': ' + 'An Error has Occured in updateISFieldsToBooleanFeatures: ' + str(e))

def removeHighCardinalityFeatures(df):
    '''
    Remove high cardinality categorical variables. High cardinality variables, when converted to dummy variables, will result in an overwhelming number of input variables, which results in a smaller number of records that can populate each level. Therefore, inferences made my modeling with dummy variables with too few records will be noisy at best and can lead to model overfit.
    Specifically, date_time, user_id, user_location_city, srch_ci, srch_co, srch_destination_id, hotel_market, user_location_region
    
    Parameters:
    df (pd.DataFrame): Input dataset from Expedia Hotel Recommendations
    
    Returns:
    df (pd.DataFrame): Converted dataFrame with the high cardinality categorical variables removed
    '''
    try:
        tic = time.perf_counter()
        
        df.drop('date_time', axis=1, inplace=True)
        df.drop('user_id', axis=1, inplace=True)
        df.drop('user_location_city', axis=1, inplace=True)
        df.drop('srch_ci', axis=1, inplace=True)
        df.drop('srch_co', axis=1, inplace=True)
        df.drop('srch_destination_id', axis=1, inplace=True)
        df.drop('hotel_market', axis=1, inplace=True)
        df.drop('user_location_region', axis=1, inplace=True)
        
        toc = time.perf_counter()

        logging.info(str(datetime.datetime.now()) + ': ' + 'removeHighCardinalityFeatures Time elapsed: '+ str(round(toc-tic, 3))+ ' seconds')

        return df
    except Exception as e:
        logging.info(str(datetime.datetime.now()) + ': ' + 'An Error has Occured in removeHighCardinalityFeatures: ' + str(e))

def removeHighNULLCntFeatures(df):
    '''
    Remove feature with an abnormally high number of missing values
    Specifically, orig_destination_distance
    
    Parameters:
    df (pd.DataFrame): Input dataset from Expedia Hotel Recommendations
    
    Returns:
    df (pd.DataFrame): Converted dataFrame with variables with a high NaN count
    '''
    
    try:
        tic = time.perf_counter()
        
        df.drop('orig_destination_distance', axis=1, inplace=True)
        
        toc = time.perf_counter()

        logging.info(str(datetime.datetime.now()) + ': ' + 'removeHighNULLCntFeatures Time elapsed: '+ str(round(toc-tic, 3))+ ' seconds')

        return df
    except Exception as e:
        logging.info(str(datetime.datetime.now()) + ': ' + 'An Error has Occured in removeHighNULLCntFeatures: ' + str(e))

def removeRemainingRecordsWithNULLS(df):
    '''
    Remove any rows containing a NULL value, just in case I missed any to this point in the DAG
    
    Parameters:
    df (pd.DataFrame): Input dataset from Expedia Hotel Recommendations
    
    Returns:
    df (pd.DataFrame): Converted dataFrame with any rows containing a NULL value removed
    '''
    
    try:
        tic = time.perf_counter()
        
        df = df.dropna(axis=0, how='any', inplace=False)
        
        toc = time.perf_counter()

        logging.info(str(datetime.datetime.now()) + ': ' + 'removeRemainingRecordsWithNULLS Time elapsed: '+ str(round(toc-tic, 3))+ ' seconds')

        return df
    except Exception as e:
        logging.info(str(datetime.datetime.now()) + ': ' + 'An Error has Occured in removeRemainingRecordsWithNULLS: ' + str(e))

def convertCategoricalVariablesToDummyVariables(df):
    '''
    Convert categorical variables into dummy variables and join to df,then drop the original variable since it is now included as dummy variables
    
    Parameters:
    df (pd.DataFrame): Input dataset from Expedia Hotel Recommendations
    
    Returns:
    df (pd.DataFrame): Converted dataFrame with categorical variables converted into dummy variables
    '''
    
    try:
        tic = time.perf_counter()
        
        dfcat = df.select_dtypes(exclude=[np.number])

        for x in dfcat:
            df = df.join(pd.get_dummies(df[x], prefix=x))
            df.drop(x, axis=1, inplace=True)
            
        toc = time.perf_counter()

        logging.info(str(datetime.datetime.now()) + ': ' + 'convertCategoricalVariablesToDummyVariables Time elapsed: '+ str(round(toc-tic, 3))+ ' seconds')

        return df
    except Exception as e:
        logging.info(str(datetime.datetime.now()) + ': ' + 'An Error has Occured in convertCategoricalVariablesToDummyVariables: ' + str(e))

def importDataset(parentPath, inputFilePath, s3=False):
    '''
    Read in dataset as a csv from the network/EC2 drive or s3 bucket.
    
    Parameters:
    parentPath (string): System generated variable. It is the root folder of the dataPipeline, relative to the user's directory.
    inputFilePath (string): User input to the location of the csv file on the network/EC2 drive or s3 bucket.
    s3 (boolean): Flag to signify if the file is located in an s3 bucket or network/EC2 drive. If located in s3 bucket, then flag should be set to True. (Default: False)
    
    Returns:
    df (pd.DataFrame): Input dataset
    '''
        
    try:
        # If user set boolean s3 flag to true, then read from s3 bucket directly
        if s3:
            tic = time.perf_counter()

            df = pd.read_csv(inputFilePath)

            toc = time.perf_counter()

            logging.info(str(datetime.datetime.now()) + ': ' + 'Import CSV Time elapsed: '+ str(round(toc-tic, 3))+ ' seconds')

            return df
        else:
            tic = time.perf_counter()

            df = pd.read_csv(parentPath+inputFilePath)

            toc = time.perf_counter()

            logging.info(str(datetime.datetime.now()) + ': ' + 'Import CSV Time elapsed: '+ str(round(toc-tic, 3))+ ' seconds')

            return df
    except Exception as e:
        logging.info(str(datetime.datetime.now()) + ': ' + 'An Error has Occured in importDataset: ' + str(e))
        
def logStartOfDataPipeline():
    '''
    Function to write to logs to signify the start of the data pipeline
    '''
    try:
        
        logging.info('########')
        logging.info(str(datetime.datetime.now()) + ': ' + 'New Data Pipeline Process')
        logging.info('########')
    except Exception as e:
        logging.info(str(datetime.datetime.now()) + ': ' + 'An Error has Occured in logStartOfApplication: ' + str(e))
        
def logEndOfDataPipeline():
    '''
    Function to write to logs to signify the end of the data pipeline
    '''
    try:
        
        logging.info('########')
        logging.info(str(datetime.datetime.now()) + ': ' + 'End of Data Pipeline Process')
        logging.info('########')
    except Exception as e:
        logging.info(str(datetime.datetime.now()) + ': ' + 'An Error has Occured in logEndOfDataPipeline: ' + str(e))
        
def exportDataset(df, outputPath, s3=False):
    '''
    Write in dataset as a csv from the network/EC2 drive or s3 bucket.
    
    Parameters:
    df (string): Cleansed dataset.
    outputPath (string): User input to write the csv file to the network/EC2 drive or s3 bucket.
    s3 (boolean): Flag to signify if the file will export an s3 bucket or network/EC2 drive. If to an s3 bucket, then flag should be set to True. (Default: False)
    
    Returns:
    csv file in the network/EC2 drive or s3 bucket
    '''
        
    try:
        # If user set boolean s3 flag to true, then write to s3 bucket directly
        if s3:
            tic = time.perf_counter()

            df.to_csv(outputPath, index=False)

            toc = time.perf_counter()

            logging.info(str(datetime.datetime.now()) + ': ' + 'Export CSV Time elapsed: '+ str(round(toc-tic, 3))+ ' seconds')

            return df
        else:
            tic = time.perf_counter()

            df.to_csv(parentPath+outputPath, index=False)

            toc = time.perf_counter()

            logging.info(str(datetime.datetime.now()) + ': ' + 'Export CSV Time elapsed: '+ str(round(toc-tic, 3))+ ' seconds')
    except Exception as e:
        logging.info(str(datetime.datetime.now()) + ': ' + 'An Error has Occured in exportDataset: ' + str(e))