INSTALLATION AND EXECUTION INSTRUCTIONS

INSTALLATION  
1. Install and configure Hadoop 2.6.0  
2. Download pre-built Apache Sparch with support of Yarn. Version: Apache Hadoop 2.6 spark 2.3.3-bin-hadoop2.6.tgz  
3. Configure ~/.bashrc file and add SPARK_HOME paths.  
4. Edit spark-defaults.conf file  
5. Start Hadoop service  
6. Clone HARP repository using:  
$ git clone https://github.com/DSC-SPIDAL/harp.git  
7. Install Maven from http://apache.spinellicreations.com/maven/maven-3/3.5.4/binaries/apache-maven-3.5.4-bin.tar.gz  
8. Compile harp  
  
EXECUTION INSTRUCTIONS  

1. Add all the data files to hadoop file system.  

HARP
1. Go to Hadoop home  
2. Run hadoop jar contrib-0.1.0.jar edu.iu.rf.RFMapCollective <no of decision trees> <no of mappers> <no of threads> <train file> <test file> <output file>  
3. Look at output in hdfs '/<output file>/output' folder  

SPARK  
1. Go to spark Home  
2. Run bin/spark-submit rf_spark.py <train file> <test file> <no of decision trees>  
3. To change number of threads, edit file and change the configuration  
Example:for 12 threads, make it local[12].  
4. Output is printed on console.  



