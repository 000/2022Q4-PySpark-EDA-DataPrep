import pyspark
from pyspark.sql import SparkSession
from operator import add
sc = pyspark.SparkContext()

spark = SparkSession(sc)


import os


#os.system('wget https://storage.googleapis.com/2022oct23/LoanStats_web.csv')
os.system('wget https://storage.cloud.google.com/2022oct23/Sony_ChurnPrediction.csv')
os.system('hdfs dfs -mkdir -p /rawzone/')
os.system('hdfs dfs -put Sony_ChurnPrediction.csv /rawzone/')


from pyspark.sql import functions as F

#raw_LendingClubWeb_df = spark.read.csv("gs://studentoct22-2/rawzone/LoanStats_web.csv", header=True, \
#                                       inferSchema=True, mode='DROPMALFORMED')


df = spark.read.format('csv').\
option('header','true').option('mode','DROPMALFORMED')\
.load('/rawzone/Sony_ChurnPrediction.csv')

import pandas as pd
import numpy as np
# set random seed to have reproducible results
# sklearn uses numpy random seed
np.random.seed(42)

df['area code'] = pd.to_numeric(df['area code'])

df_spark = spark.createDataFrame(df)

from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql.functions import when, col

# Create a new DataFrame with OneHotEncoded 'area code' column
encoder = OneHotEncoder(inputCols=["area code"], outputCols=["area_code_encoded"])
df_encoded = encoder.fit(df_spark).transform(df_spark)

# Rename the encoded columns with a prefix
#df_encoded = df_encoded.select([col(col_name).alias("area_code_" + col_name) if col_name == "area_code_encoded" else col_name for col_name in df_encoded.columns])

# Convert 'voice mail plan' and 'international plan' columns to integer type
df_encoded = df_encoded.withColumn("voice mail plan", when(df_encoded["voice mail plan"] == "no", 0).otherwise(1))
df_encoded = df_encoded.withColumn("international plan", when(df_encoded["international plan"] == "no", 0).otherwise(1))
df_encoded = df_encoded.withColumn("voice mail plan", df_encoded["voice mail plan"].cast("integer"))
df_encoded = df_encoded.withColumn("international plan", df_encoded["international plan"].cast("integer"))

# Drop unnecessary columns
df_final = df_encoded.drop("phone number", "state", "area code")

from pyspark.sql.functions import when
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create Spark DataFrame from pandas DataFrame df
df_spark = spark.createDataFrame(df)
df_spark = df_spark.withColumn("churn", df_spark["churn"].cast("integer"))


# One-hot encode area code
area_indexer = StringIndexer(inputCol='area code', outputCol='area_code_index')
area_encoder = OneHotEncoder(inputCol='area_code_index', outputCol='area_code_vec')
area_encoder.setDropLast(False)

# Convert categorical columns into numerical columns
vm_indexer = StringIndexer(inputCol='voice mail plan', outputCol='voice_mail_plan_index')
intl_indexer = StringIndexer(inputCol='international plan', outputCol='international_plan_index')
label_indexer = StringIndexer(inputCol='churn', outputCol='label')

# Assemble features vector
assembler = VectorAssembler(inputCols=['account length', 'number vmail messages', 'total day minutes', 'total day calls',
                                       'total day charge', 'total eve minutes', 'total eve calls', 'total eve charge',
                                       'total night minutes', 'total night calls', 'total night charge', 'total intl minutes',
                                       'total intl calls', 'total intl charge', 'customer service calls', 'area_code_vec',
                                       'voice_mail_plan_index', 'international_plan_index'],
                            outputCol='features')

# Split data into training and test sets
(training_data, test_data) = df_spark.randomSplit([0.8, 0.2], seed=42)

# Standardize features
scaler = StandardScaler(inputCol='features', outputCol='scaled_features')

# Train random forest model
rf = RandomForestClassifier(labelCol='label', featuresCol='scaled_features', maxDepth=5, seed=42)
rf_pipeline = Pipeline(stages=[area_indexer, area_encoder, vm_indexer, intl_indexer, label_indexer, assembler, scaler, rf])
rf_model = rf_pipeline.fit(training_data)
'''
rf_predictions = rf_model.transform(test_data)
rf_evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='f1')
rf_f1_score = rf_evaluator.evaluate(rf_predictions)
'''

rf_model.write().overwrite().save('gs://20221027-pim30-tanat-tonguthaisri-au-southeast1/refinedzone/Group3_rfModel')

# Train gradient boosted tree model
gbt = GBTClassifier(labelCol='label', featuresCol='scaled_features', maxDepth=5, seed=42)
gbt_pipeline = Pipeline(stages=[area_indexer, area_encoder, vm_indexer, intl_indexer, label_indexer, assembler, scaler, gbt])
gbt_model = gbt_pipeline.fit(training_data)
'''
gbt_predictions = gbt_model.transform(test_data)
gbt_evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='f1')
gbt_f1_score = gbt_evaluator.evaluate(gbt_predictions)
'''

gbt_model.write().overwrite().save('gs://20221027-pim30-tanat-tonguthaisri-au-southeast1/refinedzone/Group3_gbtModel')



#normalized_df.write.mode('overwrite').parquet('/rawzone/')

