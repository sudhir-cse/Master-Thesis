package de.l3s.sudhir.KMeans

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.IDF
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession

object UpdateModels {
  
/**
 * Save the model and return K clusters
 * @Input params: 
 *    inputDF - preprocessed with column name "lessImpoTermsRemoved"
 *    modelPath - path for storing models: TFModel, IDFModel and KMeans Model.
 * 
 */
def updateKMeansModel(spark:SparkSession, inputDF:Dataset[Row], k:Int, modelPath:String):Array[Dataset[Row]] = {
  
    //contains the resultant clusters
    var clusters:Array[Dataset[Row]] = new Array[Dataset[Row]](k)

    //default values, actual values will be different depending upon documents inside cluster
    val maxIter = 500
    var vocalSize = 100
    var minDF:Long = 5
    
    //compute vocalSize
    val dwcDF = inputDF.withColumn("distWordsCount", UDFs.distinctWordCount(inputDF.col("contentWords")))
    dwcDF.createOrReplaceTempView("distWordsCountModelTable")
    val avgValue = spark.sql("select avg(distWordsCount) from distWordsCountModelTable")
    vocalSize = avgValue.collect()(0).getAs[Double](0).toInt
    
    //compute minDF
    minDF = inputDF.count()/5
    
    //compute feature space
     val tfModel = new CountVectorizer().
      setInputCol("contentWords").
      setOutputCol("featuresTF").
      setVocabSize(vocalSize).
      setMinDF(minDF).
      fit(inputDF)

    //save model
    tfModel.write.overwrite().save(s"$modelPath/tfModel")
    
    val tfTransDF = tfModel.transform(inputDF)

    val idfModel = new IDF().
      setInputCol("featuresTF").
      setOutputCol("featuresTFIDF"). 
      fit(tfTransDF)
      
    //save idf model
    idfModel.write.overwrite().save(s"$modelPath/idfModel")
    
    val idfTransDF = idfModel.transform(tfTransDF)

    idfTransDF.cache()

    //KMeans
    val kmeansModel = new KMeans().
      setK(k).
      setInitMode("k-means||").
      setMaxIter(maxIter).
      setFeaturesCol("featuresTFIDF").
      setPredictionCol("clusterPrediction").
      fit(idfTransDF)
      
    //save kmeans model
    kmeansModel.write.overwrite().save(s"$modelPath/kmeansModel")
      
    val kmeansClusters = kmeansModel.transform(idfTransDF)

    //push each cluster to 'clusters' array 
    for(index <- 0 until k){
      val cluster = kmeansClusters.filter(row => row.getAs[Int]("clusterPrediction") == index)
      clusters(index) = cluster
    }

    return clusters;  
  }  
  
}