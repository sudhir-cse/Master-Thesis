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

/**
 * Provides KMeans Predictions along with distance between cluster center and each data point assigned to it 
 */
object KMeansPredictions {
  
  def performKMeansPredictions(inputDF:Dataset[Row], modelPath:String):Array[Dataset[Row]] = {
    
    //load models to perform transformation over "inputDF"
    //load first level models
    val tfModel = CountVectorizerModel.load(s"$modelPath/tfModel")
    val idfModel = IDFModel.load(s"$modelPath/idfModel")
    val kmeansModel = KMeansModel.load(s"$modelPath/kmeansModel")
    
    //perform transformations
    val tfdf = tfModel.transform(inputDF)
    val idfdf = idfModel.transform(tfdf)
    val kmeansdf = kmeansModel.transform(idfdf)
    
    val k = kmeansModel.getK
    val kmeans_centers = kmeansModel.clusterCenters.map(_.toArray)
    val resultantClusters = new Array[Dataset[Row]](k)
    
     /**
   * User defined functions [UDF]
   * computes Euclidean-squire-distance between cluster center and each data point assigned to it
   */
    val computeSQD = udf[Double, Int, SparseVector]( (clusterIndex, tfidf) => { 
        var distance: Double = 0.0
        val tfidfArray = tfidf.toArray
        val clusterCenter = kmeans_centers(clusterIndex)
        
        //compute squire distance
        if(tfidfArray.size == clusterCenter.size){
          for(index <- 0 until tfidfArray.size){
            distance = distance + (tfidfArray(index) - clusterCenter(index)) * (tfidfArray(index) - clusterCenter(index))
          }
        }   
        distance
      } 
    )
    
    //compute distance for each cluster separately
    for (index <- 0 until k){
      
      val eachCluster = kmeansdf.filter(row => row.getAs[Int]("clusterPrediction") == index)
      
      val distanceDF = eachCluster.withColumn("distance", computeSQD(eachCluster.col("clusterPrediction"), eachCluster.col("featuresTFIDF")))
      
      resultantClusters(index) = distanceDF.select("contentWords", "distance")
      
    }
   
    resultantClusters
    
  }
  
}