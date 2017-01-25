package de.l3s.sudhir.clusterInspector

import org.apache.spark.sql.SparkSession

object Result_Finisher {
  
  //Imporant variables
  //Important variables
  val INPUT_LABELS_FILE = "C:\\Users\\sudhir_itis\\Desktop\\visualization\\secondLevel\\sl_labels_0"
  val INPUT_VECTORS_FILE = "C:\\Users\\sudhir_itis\\Desktop\\visualization\\secondLevel\\sl_vectors_0"
  val OUTPUT_DIR = ""
  
  val sparkWarehouse = "temp/spark-warehouse"
    
    def main(args: Array[String]): Unit ={
      
      val spark = SparkSession.builder().
        appName("KMeans Inspector Finishes").
        master("local[*]").
        config("spark.sql.warehouse.dir", sparkWarehouse).
        getOrCreate()
      
      
      val inputLabels = spark.sparkContext.textFile(INPUT_LABELS_FILE).map(label => label.replace("[", "").replace("]", "")).zipWithIndex()
      val inputVectors = spark.sparkContext.textFile(INPUT_VECTORS_FILE).map(vector => vector.replace("[[", "").replace("]]", "")).zipWithIndex()
      
      //Convert them into DataFrame
      import spark.implicits._
      val inputLabelsDF = inputLabels.map( label => Label(label._2, label._1)).toDF()
      val inputVectorsDF = inputVectors.map(vector => Vector(vector._2, vector._1)).toDF()
      
      //Register them as a temp table
      inputLabelsDF.createOrReplaceTempView("LabelsTable")
      inputVectorsDF.createOrReplaceTempView("VectorsTable")
      
      //Join both the tables 
      val labelsVectors = spark.sql("select labels, vectors from LabelsTable lt inner join VectorsTable vt where lt.id = vt.id")
      
      //Filter out those entries that has long string.
      val filteredData = labelsVectors.filter(row => row.getAs[String]("labels").length() < 25)
      
      //filteredData.show(false)
      
      filteredData.select("labels").rdd.repartition(1).saveAsTextFile("C:\\Users\\sudhir_itis\\Desktop\\visualization\\secondLevel\\finished\\labels")
      filteredData.select("vectors").rdd.repartition(1).saveAsTextFile("C:\\Users\\sudhir_itis\\Desktop\\visualization\\secondLevel\\finished\\vectors")
      
      
      spark.stop
          
    }
  
}

//Case classes, will be used for conversation into dataframe
case class Label(id:Long, labels:String)
case class Vector(id:Long, vectors: String)  