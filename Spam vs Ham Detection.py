# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import length
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, VectorAssembler
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName('spamHam').getOrCreate()

#read data
df = spark.read.csv('/FileStore/tables/SMSSpamCollection', inferSchema = True, sep = '\t')

# COMMAND ----------

#change header
df = df.withColumnRenamed('_c0', 'class').withColumnRenamed('_c1', 'text')
df.show()

# COMMAND ----------

#length feature
df = df.withColumn('length',length(df['text']))
df.show()

# COMMAND ----------

#vectorizring text - apply tf-idf
tokenizer = Tokenizer(inputCol = 'text', outputCol='token_text')
stop_remove = StopWordsRemover(inputCol='token_text', outputCol='stop_token')
count_vec = CountVectorizer(inputCol = 'stop_token', outputCol='c_vec')
idf = IDF(inputCol='c_vec', outputCol='tf_idf')
ham_spam_to_numeric = StringIndexer(inputCol='class', outputCol='label')
cleaned_up = VectorAssembler(inputCols=['tf_idf', 'length'], outputCol='features')

# COMMAND ----------

#pipeline fit-transform
data_prep_pipe = Pipeline(stages=[ham_spam_to_numeric,tokenizer,stop_remove,count_vec,idf,cleaned_up])
cleaner = data_prep_pipe.fit(df)
clean_data = cleaner.transform(df)
clean_data = clean_data.select("label", "features")
print(clean_data.columns)
clean_data.show()
dis = clean_data.groupBy("label").count()
dis.show()

# COMMAND ----------

#train-test split
train_data, test_data = clean_data.randomSplit([0.7,0.3], seed=30)
train_data.show()
test_data.show()

# COMMAND ----------

#model generation and evaluation
model_NB = NaiveBayes()
results = model_NB.fit(train_data).transform(test_data)
#evaluation
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(results)
acc

# COMMAND ----------

#model generation and evaluation
model_NB = NaiveBayes()
results = model_NB.fit(train_data).transform(test_data)
#evaluation
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(results)
NB = acc

# COMMAND ----------

#model generation and evaluation
model_RF = RandomForestClassifier(numTrees=100)
results = model_RF.fit(train_data).transform(test_data)
#evaluation
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(results)
RF = acc

# COMMAND ----------

#model generation and evaluation
model_LR = LogisticRegression()
results = model_LR.fit(train_data).transform(test_data)
#evaluation
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(results)
LR = acc

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC 
# MAGIC X = ["Logistic Regression", "Naive Bayes", "Random Forest"]
# MAGIC Y = [LR, NB, RF]
# MAGIC 
# MAGIC 
# MAGIC # Make a fake dataset:
# MAGIC height = Y
# MAGIC bars = X
# MAGIC y_pos = np.arange(len(bars))
# MAGIC 
# MAGIC plt.figure(figsize=(10,10)) 
# MAGIC # Create bars
# MAGIC plt.bar(y_pos, height)
# MAGIC  
# MAGIC # Create names on the x-axis
# MAGIC plt.xticks(y_pos, bars)
# MAGIC 
# MAGIC # Show graphic
# MAGIC plt.show()

# COMMAND ----------

print(Y)

# COMMAND ----------


