package de.l3s.sudhir.KMeans

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.IDFModel
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.Seconds
import org.apache.spark.streaming.StreamingContext

import UpdateModels._
import LabelCluster._

/**
 * Construct the entire model hierarchy and also prints the clusters label
 */

object ConstructModelsHierarchy {
  
  def constructModelsHierarchy(spark:SparkSession, inputDF:Dataset[Row], modelHomeDir:String, k_fl:Int, k_sl:Int, topicLen_fl:Int, topicLen_sl:Int):Unit = {
    
    //construct first level model
    val path_model_fl = s"$modelHomeDir/firstLevelModels"
    val clusters_fl = updateKMeansModel(spark, inputDF, k_fl, path_model_fl)
    
    //construct second leve models
    for(index_flc <- 0 until k_fl){
      
      //prepare path to store models at this 
      val path_model_sl = s"$modelHomeDir/secondLevelModels/subModel$index_flc"
      
      val cluster_fl = clusters_fl(index_flc).select("contentWords")
      
      val clusters_sl = updateKMeansModel(spark, cluster_fl, k_sl, path_model_sl)
      
      //print clusters label
      println("----------------> Global and Local Topics Hierarchy <---------------------\n")
      
      val globalTopic = labelCluster(spark, cluster_fl, topicLen_fl)
      println(globalTopic)
      
      for(cluster_sl <- clusters_sl){
        
        val localTopic = labelCluster(spark, cluster_sl, topicLen_sl)
        println(s"\t$localTopic")
        
      }
      
      println("\n--------------------------------------------------------------------------")
      
    }
    
  }
}