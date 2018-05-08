# Twitter-Sentiment-Analysis

Build a Model to predict sentiment.


## Goal

The goal of this project is to build a model that can predict sentiment (positive or negative)
on tweets.


## Data

The data comprises of 1.6 million labeled tweets provided by Sentiment140.


## Project Workflow

Since doing NLP on a lot of data, I had to change the way I built models.  

<img align="center" src ="../master/Images/Workflow-v1.png" />
In the past, I could simply use a Jupyter notebook. But doing feature dimensionality reduction or model building
took hours.

<img align="center" src ="../master/Images/Workflow-v2.png" />
So had to develop another way. I use AWS S3 to store the data and Launch 2 EC2
instances. One will host a Data Manager which sole job is to display messages to the console and the
other instance is a powerful Spot instance. To communicate between the instances I used Redis.

<img align="center" src ="../master/Images/Workflow-v3.png" />
The next part of the workflow is to leverage Spark with distributed computing to speed up
parts of the model building process that takes a long time.

## Analysis

Since trying to predict sentiment on text, this is a NLP (Natural Languae Processing) problem.
I leveraged Spacy as my tokenizer and Sklearn tfidfVectorizer to perform my Bag of words Analysis
and tfidf transformation.

For the model I evaluated different models for accuarcy and speed. I eventually settled on XGBoost
and limiting the max features in the SVD step to explain a lot of variance but still produce results in
a timely manner. The best score of the model was ~0.74. But this was on a small amount of the data.
