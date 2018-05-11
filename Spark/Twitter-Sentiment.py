############## Initialize ##################

import time

# Feature Engineering
from pyspark.ml.feature import (VectorAssembler,VectorIndexer,
                                Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer, HashingTF)
from pyspark.sql.functions import length
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import preprocessor as p

# Models
from pyspark.ml.classification import GBTClassifier,RandomForestClassifier, NaiveBayes, LogisticRegression

# Pipeline
from pyspark.ml import Pipeline

# Evaluators
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

############## Data ########################

data = spark.read.csv("s3n://data-science-project-data/Twitter_Sentiment_Analysis/training.1600000.processed.noemoticon.csv")
data = data.withColumnRenamed('_c0','sentiment').withColumnRenamed('_c1','id').withColumnRenamed('_c2','date').withColumnRenamed('_c3','query_string').withColumnRenamed('_c4','user').withColumnRenamed('_c5','text')

#Drop Data only need 2 columns
data_dropped = data.select(['sentiment', 'text'])
data_dropped.show(5)

start_time = time.time()
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.HASHTAG)
def preprocess_tweet(tweet):
    return p.clean(tweet)

clean_tweets = udf(lambda text :preprocess_tweet(text), StringType())

clean_data = data_dropped.withColumn("clean_text", clean_tweets(col("text")))
#clean_data = clean_data.withColumnRenamed('sentiment','label')

clean_data.show(5)
print("-- Execution time: %s seconds ---" % (time.time() - start_time))

# Downsample to make it easier to deal with
(downsample_1,downsample_2) = clean_data.randomSplit([0.2,0.8])

# To avoid Data leakage split the data
(training,testing) = downsample_1.randomSplit([0.7,0.3])


################ Setup Transformations ###############################
start_time = time.time()

tokenizer = Tokenizer(inputCol="clean_text", outputCol="token_text")
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
hashingTF = HashingTF(inputCol="stop_tokens", outputCol="c_vec", numFeatures=10000)
#count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
sentiment_to_num = StringIndexer(inputCol='sentiment',outputCol='label')


# Vectorize
clean_up = VectorAssembler(inputCols=['tf_idf'],outputCol='features')

# Build Pipeline
data_prep_pipe = Pipeline(stages=[sentiment_to_num,tokenizer,stopremove,hashingTF,idf,clean_up])

# Call Pipeline for training and testing

#To prevent data leakage, transform the test data on the learned documents from training.
#This is like the real world where only have access to the training data.
cleaner = data_prep_pipe.fit(training)
training_cleaner = cleaner.transform(training)
testing_cleaner = cleaner.transform(testing)

# Select Clean Data
train_clean_data = training_cleaner.select(['label','features'])
test_clean_data = testing_cleaner.select(['label','features'])
print("-- Execution time: %s seconds ---" % (time.time() - start_time))

#################### Models #####################################
######### Naive Bayes ##########
start_time = time.time()
nb = NaiveBayes()

# Fit model
naive_model = nb.fit(train_clean_data)

# Evaluate the model
test_results = naive_model.transform(test_clean_data)
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting spam was: {}".format(acc))
print("-- Execution time: %s seconds ---" % (time.time() - start_time))

######## Logistic Regression ######
start_time = time.time()
# Setup Model
log_reg = LogisticRegression()
log_model = log_reg.fit(train_clean_data)

# Evaluate the model
test_results = log_model.transform(test_clean_data)
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting spam was: {}".format(acc))
print("-- Execution time: %s seconds ---" % (time.time() - start_time))

# Random RandomForest

start_time = time.time()
rfc = RandomForestClassifier()

# Train model.  This also runs the indexers.
rfc_model = rfc.fit(train_clean_data)

# Evaluate the model
test_results = rfc_model.transform(test_clean_data)
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting spam was: {}".format(acc))
print("-- Execution time: %s seconds ---" % (time.time() - start_time))



######### Gradient Boost #########
start_time = time.time()
gbt = GBTClassifier()
gbt_model = gbt.fit(train_clean_data)

# Evaluate the model
test_results = gbt_model.transform(test_clean_data)
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting spam was: {}".format(acc))
print("-- Execution time: %s seconds ---" % (time.time() - start_time))
