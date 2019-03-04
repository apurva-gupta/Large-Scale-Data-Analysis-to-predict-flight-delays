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
from pyspark.ml.feature import VectorAssembler

#from guppy import hpy
#h = hpy()
#threads = int(str(sys.argv[3]))
no_of_trees = str(sys.argv[3])
#time.sleep(10)
conf = SparkConf().setAppName("AirlineApp")
conf = (conf.setMaster('local[48]'))
  #      .set('spark.executor.memory', '200g')
  #      .set('spark.driver.memory', '200g')
  #      .set('spark.driver.maxResultSize', '200g'))

sc = SparkContext(conf = conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

trainfilename = str(sys.argv[1])
testfilename = str(sys.argv[2])
train = sqlContext.read.format('com.databricks.spark.csv').option("header", "true").option("inferschema", "true").option("mode", "DROPMALFORMED").load(trainfilename)
test = sqlContext.read.format('com.databricks.spark.csv').option("header", "true").option("inferschema", "true").option("mode", "DROPMALFORMED").load(testfilename)


cols = ['dep_hour_ofday','arr_hour_ofday','ActualElapsedTime','AirTime','Distance','DepDelay','TaxiIn','TaxiOut']

train = StringIndexer(inputCol='IsArrDelay', outputCol="indexedIsArrDelay").setHandleInvalid('skip').fit(train).transform(train)
train = VectorAssembler(inputCols=cols,outputCol="features").transform(train)

test = StringIndexer(inputCol='IsArrDelay', outputCol="indexedIsArrDelay").setHandleInvalid('skip').fit(test).transform(test)
test = VectorAssembler(inputCols=cols,outputCol="features").transform(test)

print('Building Random Forest Model..')

rf = RandomForestClassifier(labelCol="indexedIsArrDelay", featuresCol="features", numTrees=int(no_of_trees))
model = rf.fit(train)
predictions = model.transform(test)

print("Measuring test results:")

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="indexedIsArrDelay", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

evaluatorf1 = MulticlassClassificationEvaluator(labelCol="indexedIsArrDelay", predictionCol="prediction", metricName="f1")
f1 = evaluatorf1.evaluate(predictions)
print("f1 = %g" % f1)
#print (h.heap())
#evaluatorwp = MulticlassClassificationEvaluator(labelCol="indexedIsArrDelay", predictionCol="prediction", metricName="weightedPrecision")
#wp = evaluatorwp.evaluate(predictions)
#print("weightedPrecision = %g" % wp)

#evaluatorwr = MulticlassClassificationEvaluator(labelCol="indexedIsArrDelay", predictionCol="prediction", metricName="weightedRecall")
#wr = evaluatorwr.evaluate(predictions)
#print("weightedRecall = %g" % wr)
#sc.stop()
