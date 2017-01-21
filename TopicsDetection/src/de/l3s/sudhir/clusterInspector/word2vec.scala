package de.l3s.sudhir.clusterInspector

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row

object word2vec {
  
  
  def computeWord2Vec(data: Dataset[Row]): Dataset[Row] = {
    
    val word2VecModel = new Word2Vec()
      .setInputCol("contentToWords")
      .setOutputCol("result")
      .setVectorSize(300)
      .setMinCount(1)
      .fit(data)
    
    val vector = word2VecModel.getVectors
    
    //Return DataFrame with two columns: word and vector
    vector
    
  }
  
}