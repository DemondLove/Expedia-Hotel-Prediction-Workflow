import sys
import os
import inspect
import numpy as np
import pandas as pd

parentPath = '/'.join(sys.path[0].split('/')[:-1])

df = pd.read_csv(parentPath+'/data/pd___dfExpediaSample.csv')

def updateIDFieldsToCategoricalFeatures(df):
    '''
    Update the ID fields to be categorical features
    Specifically, site_name, posa_continent, user_location_country, user_location_region, user_location_city, user_id, channel, srch_destination_id, srch_destination_type_id, hotel_continent, hotel_country, hotel_market
    
    Parameters:
    df (pd.DataFrame): Input dataset from Expedia Hotel Recommendations
    
    Returns:
    df (pd.DataFrame): Converted dataFrame with the ID fields containing updated datatypes
    '''
    
    df['site_name'] = df['site_name'].astype('category')
    df['posa_continent'] = df['posa_continent'].astype('category')
    df['user_location_country'] = df['user_location_country'].astype('category')
    df['user_location_region'] = df['user_location_region'].astype('category')
    df['user_location_city'] = df['user_location_city'].astype('category')
    df['user_id'] = df['user_id'].astype('category')
    df['channel'] = df['channel'].astype('category')
    df['srch_destination_id'] = df['srch_destination_id'].astype('category')
    df['srch_destination_type_id'] = df['srch_destination_type_id'].astype('category')
    df['hotel_continent'] = df['hotel_continent'].astype('category')
    df['hotel_country'] = df['hotel_country'].astype('category')
    df['hotel_market'] = df['hotel_market'].astype('category')
    
    return df

def updateIS_FieldsToBooleanFeatures(df):
    '''
    Update the boolean fields to the correct datatype
    Specifically, is_mobile, is_package, is_booking
    
    Parameters:
    df (pd.DataFrame): Input dataset from Expedia Hotel Recommendations
    
    Returns:
    df (pd.DataFrame): Converted dataFrame with the boolean fields updated to the correct datatype
    '''
    
    df['is_mobile'] = df['is_mobile'].astype('bool')
    df['is_package'] = df['is_package'].astype('bool')
    df['is_booking'] = df['is_booking'].astype('bool')
    
    return df

def updateDtTmFieldsToDatetimeFeatures(df):
    '''
    Update the datetime fields to the correct datatype
    Specifically, is_mobile, is_package, is_booking
    
    Parameters:
    df (pd.DataFrame): Input dataset from Expedia Hotel Recommendations
    
    Returns:
    df (pd.DataFrame): Converted dataFrame with the datetime fields updated to the correct datatype
    '''
    
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['srch_ci'] = pd.to_datetime(df['srch_ci'])
    df['srch_co'] = pd.to_datetime(df['srch_co'])
    
    return df

def removeHighCardinalityFeatures(df):
    '''
    Remove high cardinality categorical variables. High cardinality variables, when converted to dummy variables, will result in an overwhelming number of input variables, which results in a smaller number of records that can populate each level. Therefore, inferences made my modeling with dummy variables with too few records will be noisy at best and can lead to model overfit.
    Specifically, date_time, user_id, user_location_city, srch_ci, srch_co, srch_destination_id, hotel_market, user_location_region
    
    Parameters:
    df (pd.DataFrame): Input dataset from Expedia Hotel Recommendations
    
    Returns:
    df (pd.DataFrame): Converted dataFrame with the high cardinality categorical variables removed
    '''
    
    df.drop('date_time', axis=1, inplace=True)
    df.drop('user_id', axis=1, inplace=True)
    df.drop('user_location_city', axis=1, inplace=True)
    df.drop('srch_ci', axis=1, inplace=True)
    df.drop('srch_co', axis=1, inplace=True)
    df.drop('srch_destination_id', axis=1, inplace=True)
    df.drop('hotel_market', axis=1, inplace=True)
    df.drop('user_location_region', axis=1, inplace=True)
    
    return df

def removeHighNULLCntFeatures(df):
    '''
    Remove feature with an abnormally high number of missing values
    Specifically, orig_destination_distance
    
    Parameters:
    df (pd.DataFrame): Input dataset from Expedia Hotel Recommendations
    
    Returns:
    df (pd.DataFrame): Converted dataFrame with variables with a high NaN count
    '''
    
    df.drop('orig_destination_distance', axis=1, inplace=True)
    
    return df

def removeRemainingRecordsWithNULLS(df):
    '''
    Remove any rows containing a NULL value, just in case I missed any to this point in the DAG
    
    Parameters:
    df (pd.DataFrame): Input dataset from Expedia Hotel Recommendations
    
    Returns:
    df (pd.DataFrame): Converted dataFrame with any rows containing a NULL value removed
    '''
    
    df = df.dropna(axis=0, how='any', inplace=False)
    
    return df

def convertCategoricalVariablesToDummyVariables(df):
    '''
    Convert categorical variables into dummy variables and join to df,then drop the original variable since it is now included as dummy variables
    
    Parameters:
    df (pd.DataFrame): Input dataset from Expedia Hotel Recommendations
    
    Returns:
    df (pd.DataFrame): Converted dataFrame with categorical variables converted into dummy variables
    '''
    
    dfcat = df.select_dtypes(exclude=[np.number])

    for x in dfcat:
        df = df.join(pd.get_dummies(df[x], prefix=x))
        df.drop(x, axis=1, inplace=True)

    return df
