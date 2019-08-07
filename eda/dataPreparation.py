import sys
import os
import inspect
import numpy as np
import pandas as pd

parentPath = '/'.join(sys.path[0].split('/')[:-1])

df = pd.read_csv(parentPath+'/data/pd___dfExpediaSample.csv')

def updateIDFieldsToCategoricalFeatures(dataFrame):
    '''
    Update the ID fields to be categorical features
    Specifically, site_name, posa_continent, user_location_country, user_location_region, user_location_city, user_id, channel, srch_destination_id, srch_destination_type_id, hotel_continent, hotel_country, hotel_market
    
    Parameters:
    dataFrame (pd.DataFrame): Input dataset from Expedia Hotel Recommendations
    
    Returns:
    dataFrame (pd.DataFrame): Converted dataFrame with the ID fields containing updated datatypes
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
    
# Update the boolean fields ***Data Prep Step***
df['is_mobile'] = df['is_mobile'].astype('bool')
df['is_package'] = df['is_package'].astype('bool')
df['is_booking'] = df['is_booking'].astype('bool')

# Update the DtTm fields ***Data Prep Step***
df['date_time'] = pd.to_datetime(df['date_time'])
df['srch_ci'] = pd.to_datetime(df['srch_ci'])
df['srch_co'] = pd.to_datetime(df['srch_co'])

# Remove the features outlined above ***Data Prep Step***
df.drop('date_time', axis=1, inplace=True)
df.drop('user_id', axis=1, inplace=True)
df.drop('user_location_city', axis=1, inplace=True)
df.drop('srch_ci', axis=1, inplace=True)
df.drop('srch_co', axis=1, inplace=True)
df.drop('srch_destination_id', axis=1, inplace=True)
df.drop('hotel_market', axis=1, inplace=True)
df.drop('user_location_region', axis=1, inplace=True)

# Remove feature with a high number of missing variables ***Data Prep Step***
df.drop('orig_destination_distance', axis=1, inplace=True)

# Drop any rows containing a NULL value, just in case I missed any
df = df.dropna(axis=0, how='any', inplace=False)

# Convert categorical variables into dummy variables and join to df,then drop the original variable since it is now included as dummy variables
dfcat = df.select_dtypes(exclude=[np.number])

for x in dfcat:
    df = df.join(pd.get_dummies(df['site_name'], prefix='site_name'))
    df.drop('site_name', axis=1, inplace=True)

k = ['site_name', 'posa_continent', 'user_location_country', 'is_mobile',
       'is_package', 'channel', 'srch_destination_type_id', 'is_booking',
       'hotel_continent', 'hotel_country', 'hotel_cluster']