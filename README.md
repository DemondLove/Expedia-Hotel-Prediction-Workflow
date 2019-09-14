# Expedia-Hotel-Prediction-Workflow

Workflow created ro test the processing of a large dataset, using Spark and Amazon Web Services.

We will be using the "Expedia Hotel Recommendations" Kaggle dataset: https://www.kaggle.com/c/expedia-hotel-recommendations/overview

Due to the high-volumes of data that we need to work with, we need a computing engine that will allow us to prototype and train our model on large sample batches of data, over 300,000 records on standalone machines, then seamlessly deploy the in a cloud-computing environment, which will allow us to scale our processing horizontally for much larger batches of data, over 35,000,000 records at a time. Therefore, we will be utilizing Apache Spark to handle this workflow. Apache Spark is a unified computing engine and a set of libraries for parallel data processing. Spark is designed to support a wide range of data analytics tasks, ranging from simple data loading and SQL queries to machine learning and streaming computation, over the dame computing engine and with a consistent set of APIs.

The ideal machine learning processing pipeline for this type of project, and the one we will be implementing here, is to develop a multivariate classification algorithm on a distributed Spark cluster of AWS EC2 instances, consisting of an ensemble model of Decision Tree, Logistic Regression, and Random Forest algorithms in Spark's MLlib library, to predict which hotel a user will book with data from customer behavior logs pulled from an AWS S3 bucket in CSV format, then export the predictions to an AWS S3 bucket in Parquet format.

## Data Intensive App Framework:

As with all data intensive applications, we need to develop a five stage framework, consisting of an infrastructure layer, persistence layer, integration layer, analytics layer, and engagement layer. We will be utilizing the following tools for each layer:

Infrastructure Layer: Amazon Web Services (AWS)

Persistence Layer: AWS S3 Buckets - CSV Files

Integration Layer: PySparkSQL

Analytics Layer: PySpark - MLlib

Engagement Layer: AWS S3 Buckets - Parquet Files

## Data Preparation Directed Acyclic Graph (DAG) [Integration Layer]

Functions can be found in the src/pandasDataPreparation.py

Import dataset from Presistence Layer --> updateIDFieldsToCategoricalFeatures --> removeHighCardinalityFeatures --> removeHighNULLCntFeatures --> removeRemainingRecordsWithNULLS --> convertCategoricalVariablesToDummyVariables --> Export dataset into the Analytics Layer
