############################### Imports ##################################

# Basic
import numpy as np
import pandas as pd
import scipy
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns

import boto3
import io
import warnings
warnings.filterwarnings('ignore')
import os
import redis

# NLP
import nltk
import spacy
spacy.load('en')
from nltk.corpus import stopwords
import preprocessor as p

# Model Infrastructure
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# Models
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

################################### Setup Redis ###########################

# Get Environment Variables
REDIS_IP = os.environ['REDIS_IP']

# Connect to Redis-DataStore
REDIS = redis.Redis(host=REDIS_IP)

################################## Log Function ###########################

# Code to log to the event queue


def send_event(message):
    payload = json.dumps(message)
    REDIS.publish('event_queue', payload)


#################################### Bring in Data #############################################
start_time = time.time()
s3 = boto3.client('s3')

# Bring in Training Data
obj = s3.get_object(Bucket='data-science-project-data',
                    Key='Twitter_Sentiment_Analysis/training.1600000.processed.noemoticon.csv')
cols = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
tweets = pd.read_csv(io.BytesIO(obj['Body'].read()), header=None, names=cols, encoding="ISO-8859-1")

send_event("Bring in data- Execution time: %s seconds ---" % (time.time() - start_time))


# Just Need the Sentiment and the Text
tweets.drop(['id', 'date', 'query_string', 'user'], axis=1, inplace=True)

# Clean the tweets
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.HASHTAG)


def preprocess_tweet(tweet):
    return p.clean(tweet)


# Clean the tweets, by removing special characters
start_time = time.time()
tweets['Clean'] = tweets['text'].apply(lambda x: preprocess_tweet(x))
send_event("Clean Tweets- Execution time: %s seconds ---" % (time.time() - start_time))

# Down Sample
tweets_subsampled_1, tweets_subsampled_2 = train_test_split(tweets, test_size=0.1)

# Split between outcome and Features
y = tweets_subsampled_2['sentiment']
X = tweets_subsampled_2['Clean']

# Base Model Evaluation
start_time = time.time()
# Create lemmatizer using spacy
lemmatizer = spacy.lang.en.English()


def custom_tokenizer(doc):
    tokens = lemmatizer(doc)
    return([token.lemma_ for token in tokens if not token.is_punct])


pipe = Pipeline(steps=[('vectidf', TfidfVectorizer(tokenizer=custom_tokenizer, stop_words='english',
                                                   lowercase=True, use_idf=True, max_df=0.5,
                                                   min_df=2, norm='l2', smooth_idf=True)),
                       ('svd', TruncatedSVD(500)),
                       #('norm',Normalizer(copy=False))
                       ])

tweets_transform = pipe.fit_transform(X)

send_event("Base Model- Execution time: %s seconds ---" % (time.time() - start_time))
