# Expedia-Hotel-Prediction-Workflow

Workflow created to test the processing of a large dataset, using Spark and Amazon Web Services.

We will be using the "Expedia Hotel Recommendations" Kaggle dataset: https://www.kaggle.com/c/expedia-hotel-recommendations/overview

Due to the high-volumes of data that we need to work with, we need a computing engine that will allow us to prototype and train our model on large sample batches of data, over 1,000,000 records at a time, on standalone machines, then seamlessly deploy the in a cloud-computing environment, which will allow us to scale our processing horizontally for much larges batches of data, over 35,000,000 records at a time. Apache Spark is a tool that can handle this workflow. Apache 		Spark is a unified computing engine and a set of libraries for parallel data processing. Spark is designed to support a wide range of data analytics tasks, ranging from simple data loading and SQL queries to machine learning and streaming computation, over the dame computing engine and with a consistent set of APIs.

The ideal machine learning processing pipeline is to develop a multivariate classification algorithm on a distributed Spark cluster of AWS EC2 instances, consisting of a hybrid of principle component analysis and deep learning, to predict which hotel a user will book with data from customer behavior logs pulled from an AWS RDS, then output predictions to an AWS S3 bucket. As with all data intensive applications, we need to develop a five stage framework, consisting of an infrastructure layer, persistence layer, integration layer, analytics layer, and engagement layer:

