import pyspark
from pyspark.sql import SparkSession
from operator import add
sc = pyspark.SparkContext()

spark = SparkSession(sc)


from pyspark.sql import functions as sparkf

from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, \
OneHotEncoder, VectorIndexer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, \
BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, GBTClassifier, \
NaiveBayes, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.clustering import *

normalized_df = spark.read.parquet('/rawzone/*')

labelIndexer = StringIndexer(inputCol='loan_status',outputCol='indexedLabel')

gradeIndexer = StringIndexer(inputCol='grade',outputCol='gradeIndexed')
gradeOneHotEncoder = OneHotEncoder(dropLast=False,inputCol='gradeIndexed',\
                                  outputCol='gradeVec')

homeIndexer = StringIndexer(inputCol='home_ownership',outputCol='homeIndexed')
homeOneHotEncoder = OneHotEncoder(dropLast=False,inputCol='homeIndexed',\
                                  outputCol='homeVec')

purposeIndexer = StringIndexer(inputCol='purpose',outputCol='purposeIndexed')
purposeOneHotEncoder = OneHotEncoder(dropLast=False,inputCol='purposeIndexed',\
                                  outputCol='purposeVec')

emp_lengthIndexer = StringIndexer(inputCol='emp_length',outputCol='emp_lengthIndexed')
emp_lengthOneHotEncoder = OneHotEncoder(dropLast=False,inputCol='emp_lengthIndexed',\
                                  outputCol='emp_lengthVec')

verification_statusIndexer = StringIndexer(inputCol='verification_status',outputCol='verification_statusIndexed')
verification_statusOneHotEncoder = OneHotEncoder(dropLast=False,inputCol='verification_statusIndexed',\
                                  outputCol='verification_statusVec')

featureAssembler = VectorAssembler(inputCols=['annual_inc'\
                                              #,'bc_util'\
                                              #,'inq_fi'\
                                              #,'inq_last_12m'\
                                              #,'home_ownership'\
                                              #,'purpose'\
                                              #,'emp_length'\
                                              ,'installment'\
                                              #,'total_rev_hi_lim'\
                                              ,'loan_amnt'\
                                              #,'loan_status'\
                                              #,'verification_status'\
                                              ,'total_pymnt'\
                                              ,'gradeVec'\
                                              ,'homeVec'\
                                              ,'emp_lengthVec'\
                                              ,'purposeVec'\
                                              ,'verification_statusVec']\
                                   ,outputCol='***features')

selected_attr_list = ['annual_inc'\
                                              #,'bc_util'\
                                              #,'inq_fi'\
                                              #,'inq_last_12m'\
                                              #,'home_ownership'\
                                              #,'purpose'\
                                              #,'emp_length'\
                                              ,'installment'\
                                              #,'total_rev_hi_lim'\
                                              ,'loan_amnt'\
                                              #,'loan_status'\
                                              #,'verification_status'\
                                              ,'total_pymnt'\
                                              ,'grade'\
                                              ,'home'\
                                              ,'emp_length'\
                                              ,'purpose'\
                                              ,'verification_status']

dt = DecisionTreeClassifier(featuresCol='***features',labelCol='indexedLabel')

pipeline_dt = Pipeline().setStages([gradeIndexer,gradeOneHotEncoder,\
                                    homeIndexer,homeOneHotEncoder,\
                                    emp_lengthIndexer,emp_lengthOneHotEncoder,\
                                    purposeIndexer,purposeOneHotEncoder,\
                                    verification_statusIndexer,verification_statusOneHotEncoder,\
                                    labelIndexer,\
                                    featureAssembler,\
                                    dt])

training_dt, test_dt = normalized_df.randomSplit([0.6,0.4], seed = 13)

model_dt = pipeline_dt.fit(training_dt)

model_dt.write().overwrite().save('gs://20221027-pim30-tanat-tonguthaisri-au-southeast1/refinedzone/PIM30_Tanat_Tonguthaisri_classificationModel')

