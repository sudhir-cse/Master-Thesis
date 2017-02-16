package de.l3s.sudhir.KMeans

//All the import statements
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

import UDFs._
import LabelCluster._
import UpdateModels._
import Utilities._
import KMeansPredictions._
import ConstructModelsHierarchy._

object StreamingKMeans {
  
  //clusters
  val k_fl = 4
  val topicLen_fl = 5
  val threshold_fl = 0.5
  
  val k_sl = 2
  val topicLen_sl = 4
  val threshold_sl = 0.3
  
  val sparkWarehouse = "temp/spark-warehouse"    //spark sql tables
  val ModelHomeDir = "/home/sudhir/modelHomeDir"  //home directorie for all the ML Models
  val BATCH_INTERVAL = 2
  

  def streaming(args:Array[String]):Unit = {
    
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("Topics detection in Text Stream")
      .config("spark.sql.warehouse.dir", sparkWarehouse)
      .config("spark.streaming.unpersist", "false") 
      .getOrCreate()
      
    //create an empty Dataframe
    import spark.implicits._
    var emptyDF = spark.emptyDataset[Record].toDF()
    
    //holds records for building initial model
    var initialModelDF = emptyDF
    val initialTrainingRecords = 500
    var initialModelFlag = true
    
    //global container
    var global_container = emptyDF
    val global_container_maxSize = 100 //after this limit container gets overflow
    
    //These Dataframes will be used for re-computing global model
    var global_wareHouse = emptyDF 
    val global_wareHouse_maxSize = 15000
    
    //local containers
    var localContainers = new Array[Dataset[Row]](k_fl)
    var local_containers = localContainers.map(ds => emptyDF)
    val local_container_maxSize = 100
    
    //used during computing local models
    var local_wareHouses_temp = new Array[Dataset[Row]](k_fl)
    var local_wareHouses = local_wareHouses_temp.map(ds => emptyDF)
    val local_wareHouses_maxSize = 1000
    
    //Now start streaming
    val ssc = new StreamingContext(spark.sparkContext, Seconds(BATCH_INTERVAL))
    
    val inputDStream = ssc.socketTextStream("localhost", 2222, StorageLevel.MEMORY_AND_DISK_SER)
    
    //extract file content
    val textDStream = inputDStream.map(record => {
      
      val content = parse(record, "content")
      content
      
    })
    
    //process each records
    textDStream.foreachRDD(rdd =>{
      
      //convert to DataFrame
      val tempDF = rdd.map(text => (text)).toDF("content")
      
      //split file content into tokens
      val tokenizerDF = new Tokenizer().setInputCol("content").setOutputCol("words").transform(tempDF)
      
      //remove stop words
      val stopWordsRemovedDF = tokenizerDF.withColumn("contentWords", filterStopWords(tokenizerDF.col("words")))
      
      //remove common and low document-freq terms. This is being done for the experiment
      //perform further processing as needed
      
      
      //keep only 'contentWords' column for working with
      val mainDF = stopWordsRemovedDF.select("contentWords")
      
      //Build initial models hierarchies, accumulate documents arrived at time point t0.
      if(initialModelFlag){
        
        if(initialModelDF.count() < initialTrainingRecords){
          
            initialModelDF = initialModelDF.union(mainDF)
        }
        else{
          
          initialModelFlag = false
          
          //store theses document set into warehouse and will be used during model re-construction
          global_wareHouse = global_wareHouse.union(mainDF)
          
          //construct initial models hierarchy
          constructModelsHierarchy(spark, mainDF, ModelHomeDir, k_fl, k_sl, topicLen_fl, topicLen_sl)
          
          println("Initial models were generated successfully......")
          
        }
        
      }
      
      //Process records arrived after time point t0.
      else{
        
        //perform transformation on the new documents
        val model_Path_fl =  s"$ModelHomeDir/firstLevelModels"
        val clusters_fl = performKMeansPredictions(mainDF, model_Path_fl)
        
        //try to merge new documents to existing hierarchy
        for (cluster_fl <- clusters_fl){
          
          /*
           * CASE 1: if documents do not fit into any of the first level topics then,
           * 		-they need to be stored into global container, we called them novel documents, as they are bringing new informations w.r.t globally
           */
          val novelDocs_fl = cluster_fl.filter(row => row.getAs[Int]("distance") > threshold_fl ) 
          global_container = global_container.union(novelDocs_fl.select("contentWords"))
          
          //extract documents that are fitted to either of the first level clusters
          val fittedDocs_fl = cluster_fl.filter(row => row.getAs[Int]("distance") <= threshold_fl )
          
          //perform second level of transformation to find fits to any of sub cluster
          val cluster_index_fl = clusters_fl.indexOf(cluster_fl)
          val model_path_sl = s"$ModelHomeDir/secondLevelModels/subModel$cluster_index_fl"
          val clusters_sl = performKMeansPredictions(cluster_fl, model_path_sl)
          
          for (cluster_sl <- clusters_sl){
          /*
           * CASE 2: 
           * -if documents fit to one of the first level cluster then,
           * 		-but if it does not fit to any of corresponding sub clusters then,
           * 				-store it into corresponding local containers, they are novel documents as they bring new informations important to locally
           */  
           val novelDocs_sl = cluster_sl.filter(row => row.getAs[Int]("distance") > threshold_sl )
           local_containers(cluster_index_fl) = local_containers(cluster_index_fl).union(novelDocs_sl.select("contentWords"))
           
           /*
           * CASE 3: 
           * -if documents fit to one of the first level cluster then,
           * 		-if it fits to any of corresponding sub clusters then,
           * 				-store it into corresponding local warehouse, they will be use full during reconstruction of a part of hierarchy
           */
           val fittedDocs_sl = cluster_sl.filter(row => row.getAs[Int]("distance") <= threshold_sl )
           local_wareHouses(cluster_index_fl) = local_wareHouses(cluster_index_fl).union(fittedDocs_sl.select("contentWords"))
           
          }         
          
        }
        
        /*
         * time to work with - STREAM NOVELTY
         * 
         */
        //re-construct the entire hierarchy in the following situation
        if(global_container.count() >= global_container_maxSize){
          
          val mergeLocalWareHouses = local_wareHouses.reduce((df1,df2) => df1.select("contentWords").union(df2.select("contentWords")))
          val gcontainer_localWareHouses = global_container.select("contentWords").union(mergeLocalWareHouses)
          
          constructModelsHierarchy(spark, gcontainer_localWareHouses, ModelHomeDir, k_fl, k_sl, topicLen_fl, topicLen_sl)
          
          //clear up global container
          global_container = emptyDF
          
        }
        
        //re-adjust only affected part of hierarchy
        for(local_container <- local_containers){
          
          if(local_container.count() >= local_container_maxSize){
            
            val index_local_container = local_containers.indexOf(local_container)
            val local_ware_house = local_wareHouses(index_local_container)
            val mergeContainerWarehouse = local_container.select("contentWords").union(local_ware_house.select("contentWords"))
            
            val mPath = s"$ModelHomeDir/secondLevelModels/subModel$index_local_container"
            updateKMeansModel(spark, mergeContainerWarehouse, k_sl, mPath)
            
            //clear up this local container
            local_containers(index_local_container) = emptyDF
          }
          
        }
         
      } //end of else
      
    }) //end of foreachRDD
     
  }//end of main
   
}

/** Case class for DF ("timeStamp fileName contentToWords ") */
case class Record(contentWords: Array[String])

























