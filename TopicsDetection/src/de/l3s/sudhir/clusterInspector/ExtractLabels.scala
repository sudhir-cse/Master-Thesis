package de.l3s.sudhir.clusterInspector

import org.apache.spark.sql.SparkSession

object ExtractLabels {
  
  //Imporant variables
  //Important variables
  val INPUT_LABELS_FILE = "C:\\Users\\sudhir_itis\\Desktop\\visualization\\collection\\labels"
  val INPUT_VECTORS_FILE = "C:\\Users\\sudhir_itis\\Desktop\\visualization\\collection\\vectors"
  //val OUTPUT_DIR = "C:\\Users\\sudhir_itis\\Desktop\\visualization\\vocabularies\\vocal\\vocal"
  
  val VOCALS_FILE = "C:\\Users\\sudhir_itis\\Desktop\\visualization\\vocabularies\\vocal\\vocal";
  
  val sparkWarehouse = "temp/spark-warehouse"
  
  def main(args: Array[String]): Unit = {
    
    val spark = SparkSession.builder().
        appName("Extracts Labels").
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
      filteredData.createOrReplaceTempView("filteredDataTable")
      
      //filter out only the vocal records
      val vocals = spark.sparkContext.textFile(VOCALS_FILE).collect()
      val finishVocals = vocals.map(labels => labels.replace("[", "").replace("]", ""))
      
      //finishVocals.foreach(println)
      
      //filteredData.show()
      //vocals.foreach(println)
      
     val vocaldDF = filteredData.filter(row => finishVocals contains row.getAs[String]("labels"))
      
      
      //filteredData.show(false)
      
      vocaldDF.select("labels").rdd.repartition(1).saveAsTextFile("C:\\Users\\sudhir_itis\\Desktop\\visualization\\vocabularies\\processed\\labels")
      vocaldDF.select("vectors").rdd.repartition(1).saveAsTextFile("C:\\Users\\sudhir_itis\\Desktop\\visualization\\vocabularies\\processed\\vectors")
      
      println("Done!!")
      
      spark.stop
    
  }
  
}

case class Vocal(vocals: String)