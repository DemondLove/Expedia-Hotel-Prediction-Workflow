%pyspark

# Load Packages
import pandas as pd
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.sql.functions import monotonically_increasing_id
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession

# Read in the train dataset from s3 bucket in CSV format
dfExpedia = spark.read.load(
  's3://expedia-hotel-recommendations-workflow/pd_dfExpediaSample.csv',
  format="csv",
  sep=",",
  inferSchema=True,
  header=True
)

# Expose Spark DataFrame as SQL Table
dfExpedia.createOrReplaceTempView('dfExpedia')

# Data Wrangling of the input dataset. Specifically:
# Update the ID fields to be categorical features
# Update the boolean fields to be categorical features
# Remove high cardinality categorical variables.
# Remove feature with an abnormally high number of missing values
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

# Remove any rows containing a NULL value, just in case I missed any to this point (most NULLS were consentrated to dropped columns, so there shouldn't be too many records dropped here)
df = df.na.drop()

# Prepare each categorical feature for inclusion as input into the machine learning models be completing the following on each: Index the feature (map strings to different numerical IDs and attach metadata to the DataFrame that specify what inputs correspond to what outputs) and One-Hot Encode the feature (convert each distinct value to a Boolean flag as a component in a vector)

# Prepare site_name for inclusion as input into machine learning models. 
lblIndxr = StringIndexer().setInputCol('site_name').setOutputCol('site_nameIndxr')
idxRes = lblIndxr.fit(df).transform(df)
idxRes = idxRes.drop('site_name')

ohe = OneHotEncoder().setInputCol('site_nameIndxr').setOutputCol('site_nameFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('site_nameIndxr')

# Prepare posa_continent for inclusion as input into machine learning models. 
lblIndxr = StringIndexer().setInputCol('posa_continent').setOutputCol('posa_continentIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('posa_continent')

ohe = OneHotEncoder().setInputCol('posa_continentIndxr').setOutputCol('posa_continentFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('posa_continentIndxr')

# Prepare user_location_country for inclusion as input into machine learning models. 
lblIndxr = StringIndexer().setInputCol('user_location_country').setOutputCol('user_location_countryIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('user_location_country')

ohe = OneHotEncoder().setInputCol('user_location_countryIndxr').setOutputCol('user_location_countryFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('user_location_countryIndxr')

# Prepare is_mobile for inclusion as input into machine learning models. 
lblIndxr = StringIndexer().setInputCol('is_mobile').setOutputCol('is_mobileIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('is_mobile')

ohe = OneHotEncoder().setInputCol('is_mobileIndxr').setOutputCol('is_mobileFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('is_mobileIndxr')

# Prepare is_package for inclusion as input into machine learning models. 
lblIndxr = StringIndexer().setInputCol('is_package').setOutputCol('is_packageIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('is_package')

ohe = OneHotEncoder().setInputCol('is_packageIndxr').setOutputCol('is_packageFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('is_packageIndxr')

# Prepare channel for inclusion as input into machine learning models. 
lblIndxr = StringIndexer().setInputCol('channel').setOutputCol('channelIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('channel')

ohe = OneHotEncoder().setInputCol('channelIndxr').setOutputCol('channelFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('channelIndxr')

# Prepare srch_destination_type_id for inclusion as input into machine learning models. 
lblIndxr = StringIndexer().setInputCol('srch_destination_type_id').setOutputCol('srch_destination_type_idIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('srch_destination_type_id')

ohe = OneHotEncoder().setInputCol('srch_destination_type_idIndxr').setOutputCol('srch_destination_type_idFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('srch_destination_type_idIndxr')

# Prepare hotel_continent for inclusion as input into machine learning models. 
lblIndxr = StringIndexer().setInputCol('hotel_continent').setOutputCol('hotel_continentIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('hotel_continent')

ohe = OneHotEncoder().setInputCol('hotel_continentIndxr').setOutputCol('hotel_continentFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('hotel_continentIndxr')

# Prepare hotel_country for inclusion as input into machine learning models. 
lblIndxr = StringIndexer().setInputCol('hotel_country').setOutputCol('hotel_countryIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('hotel_country')

ohe = OneHotEncoder().setInputCol('hotel_countryIndxr').setOutputCol('hotel_countryFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('hotel_countryIndxr')

# Prepare is_booking for inclusion as input into machine learning models. 
lblIndxr = StringIndexer().setInputCol('is_booking').setOutputCol('is_bookingIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('is_booking')

ohe = OneHotEncoder().setInputCol('is_bookingIndxr').setOutputCol('is_bookingFeature')
oheRes = ohe.transform(idxRes)
oheRes = oheRes.drop('is_bookingIndxr')

# Index the target variable: hotel_cluster.
lblIndxr = StringIndexer().setInputCol('hotel_cluster').setOutputCol('hotel_clusterIndxr')
idxRes = lblIndxr.fit(oheRes).transform(oheRes)
idxRes = idxRes.drop('hotel_cluster')

# Assemble all input features into a single input vecto
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


# Rename the target variable index to label and drop old index column
dataset.createOrReplaceTempView('dataset')
dataset = sqlContext.sql('''
                             SELECT
                                 hotel_clusterIndxr AS label
                                 , features
                             FROM dataset
                         ''')
dataset = dataset.drop('hotel_clusterIndxr')

# Write the cleased dataset to an s3 bucket in parquet format
dataset.write.parquet("s3://expedia-hotel-recommendations-workflow/spark_OutputCleasedDataset.parquet")


# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = dataset.randomSplit([0.7, 0.3])

# Fit Decision Tree Algorithm
dtc = DecisionTreeClassifier(labelCol="label", featuresCol="features")
dtcm = dtc.fit(trainingData)

# Save trained Logistic Regression Model to s3 Bucket
dtcm.save('s3://expedia-hotel-recommendations-workflow/dtcm_model')

# Load Pre-Trained Logistic Regression Model
dtcModel = DecisionTreeClassificationModel.load("s3://expedia-hotel-recommendations-workflow/dtcm_model")

# Make predictions with Decision Tree model on the Test Dataset
dtcPredictions = dtcModel.transform(testData)

# Calculate and print Accuracy score for Decision Tree Algorithm
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
dtcAccuracy = evaluator.evaluate(dtcPredictions)
print("Decision Tree accuracy Error = %g" % (1.0 - dtcAccuracy))

# Calculate and print F1 score for Decision Tree Algorithm
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")
dtcF1 = evaluator.evaluate(dtcPredictions)
print("Decision Tree f1 Error = %g" % (1.0 - dtcF1))

# Calculate and print Precision score for Decision Tree Algorithm
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
dtcWeightedPrecision = evaluator.evaluate(dtcPredictions)
print("Decision Tree weightedPrecision Error = %g" % (1.0 - dtcWeightedPrecision))

# Calculate and print Recall score for Decision Tree Algorithm
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedRecall")
dtcWeightedRecall = evaluator.evaluate(dtcPredictions)
print("Decision Tree weightedRecall Error = %g" % (1.0 - dtcWeightedRecall))

# Train a RandomForest algorithm
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
rfm = rf.fit(trainingData)

# Save trained Logistic Regression Model to s3 Bucket
rfm.save('s3://expedia-hotel-recommendations-workflow/rfm_model')

# Load Pre-Trained Logistic Regression Model
rfModel = RandomForestClassificationModel.load("s3://expedia-hotel-recommendations-workflow/rfm_model")

# Make predictions with Random Forest model
rfPredictions = rfModel.transform(testData)

# Calculate and print Accuracy score for Random Forest Algorithm
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
rfAccuracy = evaluator.evaluate(rfPredictions)
print("Random Forest accuracy Error = %g" % (1.0 - rfAccuracy))

# Calculate and print F1 score for Random Forest Algorithm
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")
rfF1 = evaluator.evaluate(rfPredictions)
print("Random Forest f1 Error = %g" % (1.0 - rfF1))

# Calculate and print Precision score for Random Forest Algorithm
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
rfWeightedPrecision = evaluator.evaluate(rfPredictions)
print("Random Forest weightedPrecision Error = %g" % (1.0 - rfWeightedPrecision))

# Calculate and print Recall score for Random Forest Algorithm
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedRecall")
rfWeightedRecall = evaluator.evaluate(rfPredictions)
print("Random Forest weightedRecall Error = %g" % (1.0 - rfWeightedRecall))

# Fit the Logistic Regression Algorithm
lr = LogisticRegression(maxIter=10, regParam=0.1)
lrm = lr.fit(trainingData)

# Save trained Logistic Regression Model to s3 Bucket
lrm.save('s3://expedia-hotel-recommendations-workflow/lrm_model')

# Load Pre-Trained Logistic Regression Model
lrModel = LogisticRegressionModel.load("s3://expedia-hotel-recommendations-workflow/lrm_model")

# Make predictions with the Logistic Regression Model
lrPredictions = lrModel.transform(testData)

# Calculate and print Accuracy score for Logistic Regression Algorithm
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
lrAccuracy = evaluator.evaluate(lrPredictions)
print("Logistic Regression accuracy Error = %g" % (1.0 - lrAccuracy))

# Calculate and print F1 score for Logistic Regression Algorithm
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")
lrF1 = evaluator.evaluate(lrPredictions)
print("Logistic Regression f1 Error = %g" % (1.0 - lrF1))

# Calculate and print Precision score for Logistic Regression Algorithm
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
lrWeightedPrecision = evaluator.evaluate(lrPredictions)
print("Logistic Regression weightedPrecision Error = %g" % (1.0 - lrWeightedPrecision))

# Calculate and print Recall score for Logistic Regression Algorithm
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedRecall")
lrWeightedRecall = evaluator.evaluate(lrPredictions)
print("Logistic Regression weightedRecall Error = %g" % (1.0 - lrWeightedRecall))

# Rename the Decision Tree Prediction Column Name
dtcPredictions = dtcPredictions.selectExpr("prediction as dtcPrediction")

# Add increasing Ids, and they should be the same.
dtcPredictions = dtcPredictions.withColumn("id", monotonically_increasing_id())

# Rename the Random Forest Prediction Column Name
rfPredictions = rfPredictions.selectExpr("prediction as rfPrediction")

# Add increasing Ids, and they should be the same.
rfPredictions = rfPredictions.withColumn("id", monotonically_increasing_id())

# Rename the Logistic Regression Prediction Column Name
lrPredictions = lrPredictions.selectExpr("prediction as lrPrediction")

# Add increasing Ids, and they should be the same.
lrPredictions = lrPredictions.withColumn("id", monotonically_increasing_id())

# Join Decision Tree Predictions with Random Forest Predictions
twoModelPredictions = dtcPredictions.join(rfPredictions, "id", "inner")

# Join Logistic Regression Predictions with Random Forest/Decision Tree Predictions
threeModelPredictions = twoModelPredictions.join(lrPredictions, "id", "inner").drop("id")

# Convert Spark DataFrame to Pandas DataFrame
pd_threeModelPredictions = threeModelPredictions.toPandas()

# Create new DataFrame with the mode of each row, thus taking the most common prediction amongst the three models as the final ensemble model prediction
pd_threeModelPredictions = pd_threeModelPredictions.mode(axis=1)[0]

# Rename Pandas Series column
pd_ensemblePredictions = pd_threeModelPredictions.rename('ensemblePredictions')

# Convert the Pandas Series to a Pandas DataFrame
pd_ensemblePredictions = pd_ensemblePredictions.to_frame()

# Convert the Pandas DataFrame to a Spark DataFrame to export as Parquet
ensemblePredictions = sqlContext.createDataFrame(pd_ensemblePredictions)

# Write the final predictions of the ensemble model to an s3 bucket in parquet format
ensemblePredictions.write.parquet("s3://expedia-hotel-recommendations-workflow/finalPredictions.parquet")