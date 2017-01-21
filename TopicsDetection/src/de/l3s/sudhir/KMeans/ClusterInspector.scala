package de.l3s.sudhir.KMeans

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.PCA


object ClusterInspector {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("Cluster Inspector")
      .getOrCreate()

    // Input data: Each row is a bag of words from a sentence or document.
    val documentDF = spark.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text")

    // Learn a mapping from words to Vectors.
    val word2VecModel = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3) //Number of dimension each word should be presented with, make like 1000, in the Tensorflow example, its 5000
      .setMinCount(0) //Number of time the word should appear in the document to be included into vocabularies
      .fit(documentDF) 
    val word2vecDF = word2VecModel.getVectors
    println("Word to Vector Result")
    //word2vecDF.show(false)
    
    //PCA to reduce dimension to 2
    val pcaModel = new PCA()
    .setInputCol("vector")
    .setOutputCol("pcaFeatures")
    .setK(2)
    .fit(word2vecDF)
    
    val pcaDF = pcaModel.transform(word2vecDF)
    
    println("files are being saved")
    //pcaDF.select("word").map(row => row.getAs(1))
  
    println("PCA result")
    pcaDF.show(false)

    spark.stop()
  }
}