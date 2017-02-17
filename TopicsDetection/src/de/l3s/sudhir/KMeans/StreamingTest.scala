package de.l3s.sudhir.KMeans

import org.apache.spark.sql.SparkSession
import scala.collection.mutable.Queue
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.StreamingContextState
import org.apache.spark.streaming.Seconds

object StreamingTest {
  
  def main(args:Array[String]):Unit = {
    
    val sparkWarehouse = "temp/spark-warehouse";
    val INPUT_FILE = "./data/focused/output/single/part-00000"
    var MIN_PARTITION: Int = 20
    val BATCH_INTERVAL = 1;
    
    
    val spark = SparkSession.
      builder().
      appName("Test KMeans Streaming").
      config("spark.sql.warehouse.dir", sparkWarehouse).
      getOrCreate()
      
    val mainRDD = spark.sparkContext.textFile(INPUT_FILE, MIN_PARTITION) 
    val zipIndexRDD = mainRDD.zipWithIndex().map((t) => (t._2, t._1))
    
    val mainRDDSize = mainRDD.count()
    val offset = 50 //total number of record in each resulting rdd
    val totalRDD = mainRDDSize/offset
    
    val ssc = new StreamingContext(spark.sparkContext, Seconds(BATCH_INTERVAL))
        
    // Create the queue through which RDDs can be pushed to
    // a QueueInputDStream
    val rddQueue = new Queue[RDD[(Long, String)]]()
    
    // Create the QueueInputDStream and use it do some processing
    val inputDStream = ssc.queueStream(rddQueue)
    
    inputDStream.foreachRDD(rdd => {
        
        val line = rdd.take(1)(0)._2
        println(line)
        println("----------")
      
      }
     )
    
    ssc.start()
    
    // Create and push some RDDs into rddQueue   
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