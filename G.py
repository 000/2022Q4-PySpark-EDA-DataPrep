import pyspark
from pyspark.sql import SparkSession
from operator import add
sc = pyspark.SparkContext()

spark = SparkSession(sc)


import os


os.system('wget https://storage.googleapis.com/2022oct23/LoanStats_web.csv')
os.system('hdfs dfs -mkdir -p /rawzone/')
os.system('hdfs dfs -put LoanStats_web.csv /rawzone/')


from pyspark.sql import functions as F

#raw_LendingClubWeb_df = spark.read.csv("gs://studentoct22-2/rawzone/LoanStats_web.csv", header=True, \
#                                       inferSchema=True, mode='DROPMALFORMED')


raw_LendingClubWeb_df = spark.read.format('csv').\
option('header','true').option('mode','DROPMALFORMED')\
.load('/rawzone/LoanStats_web.csv')


#raw_LendingClubWeb_df.count()

loanPayment_df = raw_LendingClubWeb_df\
.filter((F.col('loan_status') == 'Fully Paid') | ((F.col('loan_status') =='Charged Off')))

business_df = loanPayment_df.select('annual_inc'\
                                           ,'bc_util'\
                                           ,'inq_fi'\
                                           ,'inq_last_12m'\
                                           ,'home_ownership'\
                                           ,'purpose'\
                                           ,'emp_length'\
                                           ,'revol_bal'\
                                           ,'dti'\
                                           ,'delinq_2yrs'\
                                           ,'pub_rec_bankruptcies'\
                                           ,'pub_rec'\
                                           ,'open_rv_24m'\
                                           ,'mort_acc'\
                                           ,'num_actv_bc_tl'\
                                           ,'num_actv_rev_tl'\
                                           ,'num_il_tl'\
                                           ,'num_tl_90g_dpd_24m'\
                                           ,'int_rate'\
                                           ,'inq_last_6mths'\
                                           ,'term'\
                                           ,'installment'\
                                           ,'total_rev_hi_lim'\
                                           ,'total_bal_il'\
                                           ,'total_bal_ex_mort'\
                                           ,'total_acc'\
                                           ,'tot_cur_bal'\
                                           ,'loan_amnt'\
                                           ,'loan_status'\
                                           ,'verification_status'\
                                           ,'collections_12_mths_ex_med'\
                                           ,'chargeoff_within_12_mths'\
                                           ,'il_util'\
                                           ,'last_pymnt_amnt'\
                                           #,'last_pymnt_d'\
                                           ,'out_prncp_inv'\
                                           ,'out_prncp'\
                                           ,'total_pymnt_inv'\
                                           ,'total_pymnt'\
                                           ,'grade')


import pandas as pd

no_null_df = business_df.dropna(how='any')

from pyspark.sql.functions import *

no_wedding_df = no_null_df.filter(F.col('purpose') != 'wedding')

fitmem_no_null_df = no_wedding_df.repartition(60)

cached_no_null_df = fitmem_no_null_df.cache()

from pyspark.sql.functions import udf
from pyspark.sql.types import *

def f_removepercent(origin):
    return origin.rstrip('%')

removepercent = udf(lambda x: f_removepercent(x),StringType())

def f_exrtractmonth(origin):
    return origin.split('-')[0]

exrtractmonth = udf(lambda x: f_exrtractmonth(x),StringType())

def python_treatNA(origin):
    if origin == 'n/a':
        new = 'NotEmployed'
    else:
        new = origin
    return new

treatNA = udf(lambda x: python_treatNA(x),StringType())

from pyspark.sql.functions import col

crunched_df = cached_no_null_df.\
withColumn('emp_length',treatNA(cached_no_null_df['emp_length'])).\
withColumn('int_rate',removepercent(cached_no_null_df['int_rate']).cast(DoubleType())).\
withColumn('dti',cached_no_null_df['dti'].cast(DoubleType())).\
withColumn('revol_bal',cached_no_null_df['revol_bal'].cast(DoubleType())).\
withColumn('pub_rec',cached_no_null_df['pub_rec'].cast(DoubleType())).\
withColumn('total_bal_il',cached_no_null_df['total_bal_il'].cast(DoubleType())).\
withColumn('tot_cur_bal',cached_no_null_df['tot_cur_bal'].cast(DoubleType())).\
withColumn('total_acc',cached_no_null_df['total_acc'].cast(DoubleType())).\
withColumn('total_bal_ex_mort',cached_no_null_df['total_bal_ex_mort'].cast(DoubleType())).\
withColumn('total_rev_hi_lim',cached_no_null_df['total_rev_hi_lim'].cast(DoubleType())).\
withColumn('num_actv_rev_tl',cached_no_null_df['num_actv_rev_tl'].cast(DoubleType())).\
withColumn('num_actv_bc_tl',cached_no_null_df['num_actv_bc_tl'].cast(DoubleType())).\
withColumn('num_il_tl',cached_no_null_df['num_il_tl'].cast(DoubleType())).\
withColumn('pub_rec_bankruptcies',cached_no_null_df['pub_rec_bankruptcies'].cast(DoubleType())).\
withColumn('delinq_2yrs',cached_no_null_df['delinq_2yrs'].cast(DoubleType())).\
withColumn('open_rv_24m',cached_no_null_df['open_rv_24m'].cast(DoubleType())).\
withColumn('num_tl_90g_dpd_24m',cached_no_null_df['num_tl_90g_dpd_24m'].cast(DoubleType())).\
withColumn('inq_last_6mths',cached_no_null_df['inq_last_6mths'].cast(DoubleType())).\
withColumn('bc_util',cached_no_null_df['bc_util'].cast(DoubleType())).\
withColumn('mort_acc',cached_no_null_df['mort_acc'].cast(DoubleType())).\
withColumn('inq_fi',cached_no_null_df['inq_fi'].cast(DoubleType())).\
withColumn('last_pymnt_amnt',cached_no_null_df['last_pymnt_amnt'].cast(DoubleType())).\
withColumn('out_prncp_inv',cached_no_null_df['out_prncp_inv'].cast(DoubleType())).\
withColumn('out_prncp',cached_no_null_df['out_prncp'].cast(DoubleType())).\
withColumn('total_pymnt_inv',cached_no_null_df['total_pymnt_inv'].cast(DoubleType())).\
withColumn('total_pymnt',cached_no_null_df['total_pymnt'].cast(DoubleType())).\
withColumn('il_util',cached_no_null_df['il_util'].cast(DoubleType())).\
withColumn('chargeoff_within_12_mths',cached_no_null_df['chargeoff_within_12_mths'].cast(DoubleType())).\
withColumn('collections_12_mths_ex_med',cached_no_null_df['collections_12_mths_ex_med'].cast(DoubleType())).\
withColumn('loan_amnt',cached_no_null_df['loan_amnt'].cast(DoubleType())).\
withColumn('inq_last_12m',cached_no_null_df['inq_last_12m'].cast(DoubleType())).\
withColumn('installment',cached_no_null_df['installment'].cast(DoubleType())).\
withColumn('annual_inc',cached_no_null_df['annual_inc'].cast(DoubleType()))

from pyspark.sql.functions import *
max_annual_inc = crunched_df.select(max('annual_inc')).collect()[0][0]
min_annual_inc = crunched_df.select(min('annual_inc')).collect()[0][0]

def t_annual_inc(origin):
    return ((origin-min_annual_inc)/(max_annual_inc-min_annual_inc))

n_annual_inc = udf(lambda x: t_annual_inc(x),DoubleType())

max_revol_bal = crunched_df.select(max('revol_bal')).collect()[0][0]
min_revol_bal = crunched_df.select(min('revol_bal')).collect()[0][0]

def t_revol_bal(origin):
    return ((origin-min_revol_bal)/(max_revol_bal-min_revol_bal))

n_revol_bal = udf(lambda x: t_revol_bal(x),DoubleType())

max_tot_cur_bal = crunched_df.select(max('tot_cur_bal')).collect()[0][0]
min_tot_cur_bal = crunched_df.select(min('tot_cur_bal')).collect()[0][0]

def t_tot_cur_bal(origin):
    return ((origin-min_tot_cur_bal)/(max_tot_cur_bal-min_tot_cur_bal))

n_tot_cur_bal = udf(lambda x: t_tot_cur_bal(x),DoubleType())

max_total_rev_hi_lim = crunched_df.select(max('total_rev_hi_lim')).collect()[0][0]
min_total_rev_hi_lim = crunched_df.select(min('total_rev_hi_lim')).collect()[0][0]

def t_total_rev_hi_lim(origin):
    return ((origin-min_total_rev_hi_lim)/(max_total_rev_hi_lim-min_total_rev_hi_lim))

n_total_rev_hi_lim = udf(lambda x: t_total_rev_hi_lim(x),DoubleType())

max_total_bal_ex_mort = crunched_df.select(max('total_bal_ex_mort')).collect()[0][0]
min_total_bal_ex_mort = crunched_df.select(min('total_bal_ex_mort')).collect()[0][0]

def t_total_bal_ex_mort(origin):
    return ((origin-min_total_bal_ex_mort)/(max_total_bal_ex_mort-min_total_bal_ex_mort))

n_total_bal_ex_mort = udf(lambda x: t_total_bal_ex_mort(x),DoubleType())

max_total_bal_il = crunched_df.select(max('total_bal_il')).collect()[0][0]
min_total_bal_il = crunched_df.select(min('total_bal_il')).collect()[0][0]

def t_total_bal_il(origin):
    return ((origin-min_total_bal_il)/(max_total_bal_il-min_total_bal_il))

n_total_bal_il = udf(lambda x: t_total_bal_il(x),DoubleType())

normalized_df = crunched_df.\
withColumn('annual_inc',n_annual_inc(crunched_df['annual_inc'])).\
withColumn('revol_bal',n_revol_bal(crunched_df['revol_bal'])).\
withColumn('tot_cur_bal',n_tot_cur_bal(crunched_df['tot_cur_bal'])).\
withColumn('total_rev_hi_lim',n_total_rev_hi_lim(crunched_df['total_rev_hi_lim'])).\
withColumn('total_bal_il',n_total_bal_il(crunched_df['total_bal_il'])).\
withColumn('total_bal_ex_mort',n_total_bal_ex_mort(crunched_df['total_bal_ex_mort']))

normalized_filtered_df = normalized_df

data_no_missing_df = normalized_filtered_df.dropna(how='any')

# Update to your GCS bucket
gcs_bucket = 'dataproc-bucket-name'

gcs_filepath = 'gs://studentoct22-2/refinedzone/pim30_purpose.csv'.format(gcs_bucket)

#data_no_missing_df.coalesce(1).write \
#  .mode('overwrite') \
#  .csv(gcs_filepath)


normalized_df.write.mode('overwrite').parquet('/rawzone/')

