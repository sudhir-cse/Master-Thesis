package de.l3s.sudhir.KMeans

import scala.collection.mutable.Queue

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.Seconds
import org.apache.spark.streaming.StreamingContext


//An entry point for driving KMeans Streaming
object Main {
  
  def main(args:Array[String]):Unit = {
    
    val sparkWarehouse = "temp/spark-warehouse";
     //Input and output files
    val INPUT_FILE = "./data/focused/output/single/part-00000"
    var MIN_PARTITION: Int = 20
    val BATCH_INTERVAL = 1
    
    val spark = SparkSession.
      builder().
      appName("Streaming KMeans").
      config("spark.sql.warehouse.dir", sparkWarehouse).
      getOrCreate()
      
    val mainRDD = spark.sparkContext.textFile(INPUT_FILE, MIN_PARTITION) 
    val zipIndexRDD = mainRDD.zipWithIndex().map((t) => (t._2, t._1))
    
    //will be used for splitting main RDDs
    val mainRDDSize = mainRDD.count()
    val offset = 50L //total number of record in each resulting rdd
    val totalRDD = mainRDDSize/offset
    
    val ssc = new StreamingContext(spark.sparkContext, Seconds(BATCH_INTERVAL))
        
    // Create the queue through which RDDs can be pushed to
    // a QueueInputDStream
    val rddQueue = new Queue[RDD[(Long, String)]]()
    
    // Create the QueueInputDStream and use it do some processing
    val inputDStream = ssc.queueStream(rddQueue)
    
    StreamingKMeans.kmeansStreaming(spark, inputDStream)
    
    ssc.start()
    
    // Create and push RDDs into rddQueue   
    for (i <- 0L until totalRDD){
      
      val startIndex = offset*i
      val endIndex = offset*(i+1)
      
      rddQueue.synchronized{
        
        rddQueue += zipIndexRDD.filter(t => (t._1 >= startIndex && t._1 < endIndex))   
      }
      
      Thread.sleep(1000)
      
    }
    ssc.stop()
  }
  
}