#!/usr/bin/env python
# encoding: utf-8

# Load Packages
import os
import sys
import time
import inspect
import numpy as np
import pandas as pd
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler

# Creating Spark Context
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession

# Run the first time:
sc = SparkContext("local")

sqlContext = SQLContext(sc)
spark = SparkSession.builder.getOrCreate()

########

dfExpedia = spark.read.load(
  '/data/train.csv',
  format="csv",
  sep=",",
  inferSchema=True,
  header=True
)

dfExpedia.createOrReplaceTempView('dfSQL')

########

df = sqlContext.sql('''
    SELECT
    
    cast(site_name AS STRING),
    cast(posa_continent AS STRING),
    cast(user_location_country AS STRING),
    orig_destination_distance,
    cast(is_mobile AS STRING),
    cast(is_package AS STRING),
    cast(channel AS STRING),
    srch_adults_cnt,
    srch_children_cnt,
    srch_rm_cnt,
    cast(srch_destination_type_id AS STRING),
    cast(is_booking AS STRING),
    cnt,
    cast(hotel_continent AS STRING),
    cast(hotel_country AS STRING),
    cast(hotel_cluster AS STRING)
    
    FROM dfExpedia
''')

########

df = df.na.drop()

########

lblIndxr = StringIndexer().setInputCol('site_name').setOutputCol('site_nameIndxr')
idxRes = lblIndxr.fit(df).transform(df)
idxRes = idxRes.drop('site_name')

ohe = OneHotEncoder().setInputCol('site_nameIndxr').setOutputCol('site_nameFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('site_nameIndxr')

########

lblIndxr = StringIndexer().setInputCol('posa_continent').setOutputCol('posa_continentIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('posa_continent')

ohe = OneHotEncoder().setInputCol('posa_continentIndxr').setOutputCol('posa_continentFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('posa_continentIndxr')

########

lblIndxr = StringIndexer().setInputCol('user_location_country').setOutputCol('user_location_countryIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('user_location_country')

ohe = OneHotEncoder().setInputCol('user_location_countryIndxr').setOutputCol('user_location_countryFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('user_location_countryIndxr')

########

lblIndxr = StringIndexer().setInputCol('is_mobile').setOutputCol('is_mobileIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('is_mobile')

ohe = OneHotEncoder().setInputCol('is_mobileIndxr').setOutputCol('is_mobileFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('is_mobileIndxr')

########

lblIndxr = StringIndexer().setInputCol('is_package').setOutputCol('is_packageIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('is_package')

ohe = OneHotEncoder().setInputCol('is_packageIndxr').setOutputCol('is_packageFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('is_packageIndxr')

########

lblIndxr = StringIndexer().setInputCol('channel').setOutputCol('channelIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('channel')

ohe = OneHotEncoder().setInputCol('channelIndxr').setOutputCol('channelFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('channelIndxr')

########

lblIndxr = StringIndexer().setInputCol('srch_destination_type_id').setOutputCol('srch_destination_type_idIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('srch_destination_type_id')

ohe = OneHotEncoder().setInputCol('srch_destination_type_idIndxr').setOutputCol('srch_destination_type_idFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('srch_destination_type_idIndxr')

########

lblIndxr = StringIndexer().setInputCol('hotel_continent').setOutputCol('hotel_continentIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('hotel_continent')

ohe = OneHotEncoder().setInputCol('hotel_continentIndxr').setOutputCol('hotel_continentFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('hotel_continentIndxr')

########

lblIndxr = StringIndexer().setInputCol('hotel_country').setOutputCol('hotel_countryIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('hotel_country')

ohe = OneHotEncoder().setInputCol('hotel_countryIndxr').setOutputCol('hotel_countryFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('hotel_countryIndxr')

########

lblIndxr = StringIndexer().setInputCol('is_booking').setOutputCol('is_bookingIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('is_booking')

ohe = OneHotEncoder().setInputCol('is_bookingIndxr').setOutputCol('is_bookingFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('is_bookingIndxr')

########

lblIndxr = StringIndexer().setInputCol('hotel_cluster').setOutputCol('hotel_clusterIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('hotel_cluster')

########

va = VectorAssembler().setInputCols(['orig_destination_distance'
                                     ,'srch_adults_cnt'
                                     ,'srch_children_cnt'
                                     ,'srch_rm_cnt'
                                     ,'cnt'
                                     ,'is_bookingFeature'
                                     ,'site_nameFeature'
                                     ,'posa_continentFeature'
                                     ,'user_location_countryFeature'
                                     ,'is_mobileFeature'
                                     ,'is_packageFeature'
                                     ,'channelFeature'
                                     ,'srch_destination_type_idFeature'
                                     ,'hotel_continentFeature'
                                     ,'hotel_countryFeature']).setOutputCol('features')
dataset = va.transform(idxRes)

########

