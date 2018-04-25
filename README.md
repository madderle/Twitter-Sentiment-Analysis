# Twitter-Sentiment-Analysis

Build a Model to predict sentiment.


## Goal

The goal of this project is to build a model that can predict sentiment (positive or negative)
on tweets.


## Data

The data comprises of 1.6 million labeled tweets provided by Sentiment140.


## Project Workflow

Since doing NLP on a lot of data, I had to change the way I built models. In the past, I
could simply use a Jupyter notebook. But doing feature dimensionality reduction or model building
took hours. So had to develop another way. I use AWS S3 to store the data and Launch 2 EC2
instances. One will host a Data Manager which sole job is to display messages to the console and the
other instance is a powerful Spot instance. To communicate between the instances I used Redis.
