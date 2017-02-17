package de.l3s.sudhir.KMeans

import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.IDF
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession

object LabelCluster {
  
  /**
* Method to label clusters
* inputColumen: contentToWords
*/
def labelCluster(spark:SparkSession, inputClusterDF:Dataset[Row], topicLength:Int):String = {

    var topic: String = ""

    //default values, actual values will be calculated depending upon cluster size
    var vocalSize = 100
    var minDF:Long = 5
    
    //compute vocalSize
    val dwcDF = inputClusterDF.withColumn("distWordsCount", UDFs.distinctWordCount(inputClusterDF.col("contentWords")))
    dwcDF.createOrReplaceTempView("distWordsCountLabelTable")
    val avgValue = spark.sql("select avg(distWordsCount) from distWordsCountLabelTable")
    vocalSize = avgValue.collect()(0).getAs[Double](0).toInt
    
    //compute minDF
    minDF = (inputClusterDF.count()/5)
    
    //Term-frequencies vector
    val tfModel = new CountVectorizer().
        setInputCol("contentWords").
        setOutputCol("featuresTF").
        setVocabSize(vocalSize).
        setMinDF(minDF).
        fit(inputClusterDF)

    val tfTransDF  = tfModel.transform(inputClusterDF)

    //TF-IDF vector
    val tfidfModel = new IDF().
        setInputCol("featuresTF").
        setOutputCol("featuresTFIDF").
        fit(tfTransDF)

    val vocab = tfModel.vocabulary
    val tfidfWeight  = tfidfModel.idf.toArray

    val vocabAndWeight = vocab.map { term => (term, tfidfWeight(vocab.indexOf(term))) }

    //now sort by weight
    val sortedVocabAndWeight = vocabAndWeight.sortWith((tuple1, tuple2) => tuple1._2 > tuple2._2)

    //sortedVocabAndWeight.foreach(println)

    val impoTopics = sortedVocabAndWeight.map((tuple) => tuple._1)

    //argument to take is the number of vocabularies terms used for topic
    impoTopics.take(topicLength).foreach { term => topic = topic + " "+term }

    return topic
  }
  
}