package de.l3s.sudhir.clusterInspector

import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row

object pca {
  
  def reduceDimensionTo2(data:Dataset[Row]): Dataset[Row] = {
    
    val pcaModel = new PCA()
      .setInputCol("vector")
      .setOutputCol("pcaFeatures")
      .setK(2)
      .fit(data)

    val result = pcaModel.transform(data)
    
    result
    
  }
  
}