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
  var firstLevelClustersCrenter: Array[Array[Double]] = null
  

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
    val gobal_container_maxSize = 100 //after this limit container gets overflow
    
    //These Dataframes will be used for re-computing global model
    var global_wareHouse = emptyDF 
    val global_wareHouse_maxSize = 15000
    
    //local containers
    val local_containers_temp = new Array[Dataset[Row]](k_fl)
    val localContainers = local_containers_temp.map(ds => emptyDF)
    
    //used during computing local models
    val local_wareHouses_temp = new Array[Dataset[Row]](k_fl)
    val local_wareHouses = local_wareHouses_temp.map(ds => emptyDF)
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
          
          //construct first level model
          val path_model_fl = s"$ModelHomeDir/firstLevelModels"
          val clusters_fl = updateKMeansModel(spark, mainDF, k_fl,path_model_fl)
          
          //construct second leve models
          for(index_flc <- 0 until k_fl){
            
            //prepare path to store models at this 
            val path_model_sl = s"$ModelHomeDir/secondLevelModels/subModel$index_flc"
            
            val cluster_fl = clusters_fl(index_flc).select("contentWords")
            
            val clusters_sl = updateKMeansModel(spark, cluster_fl, k_sl, path_model_sl)
            
            //print clusters label
            println("----------------Global and Local Topics Hierarchy---------------------------\n")
            
            val globalTopic = labelCluster(spark, cluster_fl, topicLen_fl)
            println(globalTopic)
            
            for(cluster_sl <- clusters_sl){
              
              val localTopic = labelCluster(spark, cluster_sl, topicLen_sl)
              println(s"\t$localTopic")
              
            }
            
            println("\n----------------------------------------------------------------------------")
            
          }
          
          println("Initial models were generated successfully......")
          
        }
        
      }
      
      //Process records arrived after time point t0.
      else{
        
        //load first level models
        val tfModel_fl = CountVectorizerModel.load(s"$ModelHomeDir/firstLevelModels/tfModel")
        val idfModel_fl = IDFModel.load(s"$ModelHomeDir/firstLevelModels/idfModel")
        val kmeansModel_fl = KMeansModel.load(s"$ModelHomeDir/firstLevelModels/kmeansModel")
        
        //perform transformations
        val tfdf_fl = tfModel_fl.transform(mainDF)
        val idfdf_fl = idfModel_fl.transform(tfdf_fl)
        val kmeansdf_fl = kmeansModel_fl.transform(idfdf_fl)
        
        //compute first level clusters centers
        val kmeans_centers_fl = kmeansModel_fl.clusterCenters
        firstLevelClustersCrenter = kmeans_centers_fl.map(_.toArray)
        
        //process each clusters separately
        for (index_flc <- 0 until k_fl ){
          
          //access each cluster
          val cluster_fl = kmeansdf_fl.filter(row => row.getAs[Int]("clusterPrediction") == index_flc)
          
          //compute distance between cluster and each data point. And select only tow columns: "contentWords" and "distance"
          val distance_fl = cluster_fl.withColumn("distance", computeSQD(cluster_fl.col("clusterPrediction"), cluster_fl.col("featuresTFIDF"))).select("contentWords", "distance")
          
          /*
           * CASE 1: if documents do not fit into any of the first level topics then,
           * 		-they need to be stored into global container, we call them novel documents, as they are bringing new informations w.r.t globally
           */
          val novelDocs_fl = distance_fl.filter(row => row.getAs[Int]("distance") > threshold_fl )
          global_container = global_container.union(novelDocs_fl.select("contentWords"))
          
          /*
           * CASE 2: 
           * -if documents fit to one of the first level cluster then,
           * 		-if it fits to one of sub cluster then,
           * 				-store it into corresponding local warehouse as this will get observed by existing hierarchy
           */
          
          //extract documents are fitted to either of the first level clusters
          val fittedDocs_fl = distance_fl.filter(row => row.getAs[Int]("distance") <= threshold_fl )
          
          //check if it fits to any of the corresponding second level clusters
          //load second level models
          val tfModel_sl = CountVectorizerModel.load(s"$ModelHomeDir/secondLevelModels/subModel$index_flc/tfModel")
          val idfModel_sl = IDFModel.load(s"$ModelHomeDir/secondLevelModels/subModel$index_flc/idfModel")
          val kmeansModel_sl = KMeansModel.load(s"$ModelHomeDir/secondLevelModels/subModel$index_flc/kmeansModel")
          
          //perform transformations
          val tfdf_sl = tfModel_sl.transform(fittedDocs_fl.select("contentWords"))
          val idfdf_sl = idfModel_sl.transform(fittedDocs_fl.select("contentWords"))
          val kmeansdf_sl = kmeansModel_sl.transform(fittedDocs_fl.select("contentWords"))
            
          
          /*
           * CASE 3: 
           * -if documents fit to one of the first level cluster then,
           * 		-if it does not fits to one of sub cluster then,
           * 				-store it into corresponding local containers, we call them novel documents as the are bringing new informations w.r.t. locally
           */
        }
        
      }
      
    }) //end of foreachRDD
     
  }//end of main
  
  /**
   * User defined functions [UDF]
   * computes Euclidean-squire-distance between cluster center and each data point assigned to it
   */
  val computeSQD = udf[Double, Int, SparseVector]( (clusterIndex, tfidf) => { 
      var distance: Double = 0.0
      val tfidfArray = tfidf.toArray
      val clusterCenter = firstLevelClustersCrenter(clusterIndex)
      
      //compute squire distance
      if(tfidfArray.size == clusterCenter.size){
        for(index <- 0 until tfidfArray.size){
          distance = distance + (tfidfArray(index) - clusterCenter(index)) * (tfidfArray(index) - clusterCenter(index))
        }
      }   
      distance
    } 
  )
  
}

/** Case class for DF ("timeStamp fileName contentToWords ") */
case class Record(contentWords: Array[String])
case class Record1(contentWords: Array[String], age:Int)
























