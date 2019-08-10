#!/usr/bin/env spark
# encoding: utf-8
import time

tic = time.perf_counter()

# Load Packages
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import monotonically_increasing_id


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
  '/home/kl/Documents/Expedia-Hotel-Prediction-Workflow/data/pd_dfExpediaSample.csv',
  format="csv",
  sep=",",
  inferSchema=True,
  header=True
)

dfExpedia.createOrReplaceTempView('dfExpedia')

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

dataset.write.parquet("/home/kl/Documents/Expedia-Hotel-Prediction-Workflow/data/spark_CleasedDataset.parquet")

toc = time.perf_counter()

print("Data Preprocessing:", round(toc-tic, 3), "seconds")

tic = time.perf_counter()

dataset.createOrReplaceTempView('dataset')

dataset = sqlContext.sql('''
                             SELECT
                                 hotel_clusterIndxr AS label
                                 , features
                             FROM dataset
                         ''')

dataset = dataset.drop('hotel_clusterIndxr')

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = dataset.randomSplit([0.7, 0.3])

dtc = DecisionTreeClassifier(labelCol="label", featuresCol="features")

dtcModel = dtc.fit(trainingData)

# Make predictions.
dtcPredictions = dtcModel.transform(testData)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
dtcAccuracy = evaluator.evaluate(dtcPredictions)
print("Decision Tree accuracy Error = %g" % (1.0 - dtcAccuracy))

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")
dtcF1 = evaluator.evaluate(dtcPredictions)
print("Decision Tree f1 Error = %g" % (1.0 - dtcF1))

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
dtcWeightedPrecision = evaluator.evaluate(dtcPredictions)
print("Decision Tree weightedPrecision Error = %g" % (1.0 - dtcWeightedPrecision))

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedRecall")
dtcWeightedRecall = evaluator.evaluate(dtcPredictions)
print("Decision Tree weightedRecall Error = %g" % (1.0 - dtcWeightedRecall))

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)

rfModel = rf.fit(trainingData)

# Make predictions.
rfPredictions = rfModel.transform(testData)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
rfAccuracy = evaluator.evaluate(rfPredictions)
print("Random Forest accuracy Error = %g" % (1.0 - rfAccuracy))

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")
rfF1 = evaluator.evaluate(rfPredictions)
print("Random Forest f1 Error = %g" % (1.0 - rfF1))

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
rfWeightedPrecision = evaluator.evaluate(rfPredictions)
print("Random Forest weightedPrecision Error = %g" % (1.0 - rfWeightedPrecision))

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedRecall")
rfWeightedRecall = evaluator.evaluate(rfPredictions)
print("Random Forest weightedRecall Error = %g" % (1.0 - rfWeightedRecall))

lr = LogisticRegression(maxIter=10, regParam=0.1)

# Fit the model
lrModel = lr.fit(trainingData)

# Make predictions.
lrPredictions = lrModel.transform(testData)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
lrAccuracy = evaluator.evaluate(lrPredictions)
print("Logistic Regression accuracy Error = %g" % (1.0 - lrAccuracy))

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")
lrF1 = evaluator.evaluate(lrPredictions)
print("Logistic Regression f1 Error = %g" % (1.0 - lrF1))

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
lrWeightedPrecision = evaluator.evaluate(lrPredictions)
print("Logistic Regression weightedPrecision Error = %g" % (1.0 - lrWeightedPrecision))

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedRecall")
lrWeightedRecall = evaluator.evaluate(lrPredictions)
print("Logistic Regression weightedRecall Error = %g" % (1.0 - lrWeightedRecall))

dtcPredictions = dtcPredictions.selectExpr("prediction as dtcPrediction")

# Add increasing Ids, and they should be the same.
dtcPredictions = dtcPredictions.withColumn("id", monotonically_increasing_id())

rfPredictions = rfPredictions.selectExpr("prediction as rfPrediction")

# Add increasing Ids, and they should be the same.
rfPredictions = rfPredictions.withColumn("id", monotonically_increasing_id())

lrPredictions = lrPredictions.selectExpr("prediction as lrPrediction")

# Add increasing Ids, and they should be the same.
lrPredictions = lrPredictions.withColumn("id", monotonically_increasing_id())

# Add increasing Ids, and they should be the same.
df3 = dtcPredictions.join(rfPredictions, "id", "inner")
df4 = df3.join(lrPredictions, "id", "inner").drop("id")

pd_df4 = df4.toPandas()

pd_df4 = pd_df4.mode(axis=1)[0]

toc = time.perf_counter()

print("ML:", round(toc-tic, 3), "seconds")
