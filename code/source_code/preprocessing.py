from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import * 
from pyspark import SparkContext,SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors, VectorUDT,DenseVector
from pyspark.mllib.util import MLUtils
import sys
from pyspark.sql import functions as f
from pyspark.sql import types as t
import time

conf = SparkConf().setAppName("AirlineApp")
conf = (conf.setMaster('local[5]')
        .set('spark.executor.memory', '200g')
        .set('spark.driver.memory', '200g')
        .set('spark.driver.maxResultSize', '200g'))

sc = SparkContext(conf = conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

df = sqlContext.read.format('com.databricks.spark.csv').option("header", "true").option("inferschema", "true").option("mode", "DROPMALFORMED").load("/data/2008.csv")

time_start = time.clock()

print (df.count(),"Number of rows")

def convertColumn(df, names, newType):
  for name in names:
     df = df.withColumn(name, df[name].cast(newType))
  return df


columns_toString = ['FlightNum','CancellationCode']
columns_toInt = ['DepTime','CRSDepTime','ArrTime','CRSArrTime','ActualElapsedTime','AirTime','ArrDelay','DepDelay','Distance','TaxiIn','TaxiOut','CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay','CRSElapsedTime']
df = convertColumn(df, columns_toString, StringType())
df = convertColumn(df,columns_toInt,FloatType())


from pyspark.ml.feature import Imputer
imputer = Imputer(inputCols=columns_toInt, outputCols=columns_toInt)
model = imputer.fit(df)
df = model.transform(df)

df = df.withColumn('dep_min_ofhour', df.DepTime.substr(-4,2))
df = df.withColumn('dep_hour_ofday', df.DepTime.substr(-6,2))
df = convertColumn(df,['dep_min_ofhour','dep_hour_ofday'],FloatType())

df = df.withColumn('crsdep_min_ofhour', df.CRSDepTime.substr(-4,2))
df = df.withColumn('crsdep_hour_ofday', df.CRSDepTime.substr(-6,2))
df = convertColumn(df,['crsdep_min_ofhour','crsdep_hour_ofday'],FloatType())

df = df.withColumn('arr_min_ofhour', df.ArrTime.substr(-4,2))
df = df.withColumn('arr_hour_ofday', df.ArrTime.substr(-6,2))
df = convertColumn(df,['arr_min_ofhour','arr_hour_ofday'],FloatType())

df = df.withColumn('crsarr_min_ofhour', df.CRSArrTime.substr(-4,2))
df = df.withColumn('crsarr_hour_ofday', df.CRSArrTime.substr(-6,2))
df = convertColumn(df,['crsarr_min_ofhour','crsarr_hour_ofday'],FloatType())

df.groupBy('dep_hour_ofday').count().show()
df.groupBy('arr_hour_ofday').count().show()


from pyspark.sql.functions import when   


df = df.na.fill({'dep_hour_ofday': 0.0, 'arr_hour_ofday': 0.0})
df.groupBy('dep_hour_ofday').count().show()
df.groupBy('arr_hour_ofday').count().show()


udf_category = udf(lambda ArrDelay : 0  if ArrDelay <= 15 else 1 , IntegerType())
df = df.withColumn("IsArrDelay",udf_category(df.ArrDelay))


cols = ['dep_hour_ofday','arr_hour_ofday','ActualElapsedTime','AirTime','Distance','DepDelay','TaxiIn','TaxiOut']

clean_df = df.select('dep_hour_ofday','arr_hour_ofday','ActualElapsedTime','AirTime','Distance','DepDelay','TaxiIn','TaxiOut','IsArrDelay')

print ("pre processing successful")

print("writing 2.5 percent data")
set1, set2 = clean_df.randomSplit([0.025, 0.975])
train,test = set1.randomSplit([0.5,0.5])
train.write.format("com.databricks.spark.csv").option("sep",",").option("header", "True").save("train_airline_spark_2p5.csv")
test.write.format("com.databricks.spark.csv").option("sep",",").option("header", "True").save("test_airline_spark_2p5.csv")

print("writing 5 percent data")
set1, set2 = clean_df.randomSplit([0.05, 0.95])
train,test = set1.randomSplit([0.5,0.5])
train.write.format("com.databricks.spark.csv").option("sep",",").option("header", "True").save("train_airline_spark_5.csv")
test.write.format("com.databricks.spark.csv").option("sep",",").option("header", "True").save("test_airline_spark_5.csv")

print("writing 10 percent data")
set1, set2 = clean_df.randomSplit([0.1, 0.9])
train,test = set1.randomSplit([0.5,0.5])
train.write.format("com.databricks.spark.csv").option("sep",",").option("header", "True").save("train_airline_spark_10.csv")
test.write.format("com.databricks.spark.csv").option("sep",",").option("header", "True").save("test_airline_spark_10.csv")

print("writing 25 percent data")
set1, set2 = clean_df.randomSplit([0.25, 0.75])
train,test = set1.randomSplit([0.5,0.5])
train.write.format("com.databricks.spark.csv").option("sep",",").option("header", "True").save("train_airline_spark_25.csv")
test.write.format("com.databricks.spark.csv").option("sep",",").option("header", "True").save("test_airline_spark_25.csv")

print("writing 50 percent data")
set1, set2 = clean_df.randomSplit([0.5, 0.5])
train,test = set1.randomSplit([0.5,0.5])
train.write.format("com.databricks.spark.csv").option("sep",",").option("header", "True").save("train_airline_spark_50.csv")
test.write.format("com.databricks.spark.csv").option("sep",",").option("header", "True").save("test_airline_spark_50.csv")

print("writing all data")
train,test = clean_df.randomSplit([0.5,0.5])
train.write.format("com.databricks.spark.csv").option("sep",",").option("header", "True").save("train_airline_spark.csv")
test.write.format("com.databricks.spark.csv").option("sep",",").option("header", "True").save("test_airline_spark.csv")

print("done!")


