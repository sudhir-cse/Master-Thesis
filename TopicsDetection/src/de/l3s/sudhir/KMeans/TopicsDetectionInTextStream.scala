/*
 * Final one
 */
package de.l3s.sudhir.KMeans

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.clustering.KMeansSummary
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.IDFModel
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.Seconds
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.ml.linalg.SparseVector


object TopicsDetectionInTextStream {
  
  //val data = "data/KDDTraining" //data source
  //val data = "data/news/uk/UK_1K"
  //val data = "data/UK_news/test"
  
  //KMeans setting parameters
  val firstLevelClustersNum: Int = 3
  val firstLevelClusterMaxItr: Int = 100
  val firstLevelVocalSize: Int = 100
  val firstLevelMinDF: Int = 5
  
  val secondLevelClustersNum: Int = 2
  val secondLevelClusterMaxItr: Int = 100
  val secondLevelVocalSize: Int = 50
  val secondLevelMinDF: Int = 1
  
  val firstLevelThreshold: Double = 0.6
  val secondLevelThreshold: Double = 0.5
 
  val sparkWarehouse = "temp/spark-warehouse";
  
  val BATCH_INTERVAL = 2
  
  var firstLevelClustersCrenter: Array[Array[Double]] = null
  
  /** ---------------- main() ----------------  */
	  
  def main(args: Array[String]): Unit = {
   
    //System.setProperty("hadoop.home.dir", "E:\\master-thesis-workspace\\WinUtils");
    
    val spark = SparkSession.builder()
    .master("local[*]")
    .appName("Topics detection in Text Stream")
    .config("spark.sql.warehouse.dir", sparkWarehouse)
    .config("spark.streaming.unpersist", "false") 
    .getOrCreate()
    
    //variables for collecting initial few rdd for training the models
    import spark.implicits._
    var firstModelDF = spark.emptyDataset[Record].toDF()
    val initialTrainingRecords = 500
    var initialModelFlag = true
    
    //Containers
    var container_fl = spark.emptyDataset[Record].toDF()
    val container_fl_sizeLimit = 100  // after this limits, container gets overflow
    
    
    //These Dataframes will be used for re-computing models
    var wareHouse_fl = spark.emptyDataset[Record].toDF()
    val wareHouse_fl_sizeLimit = 15000
    
    var wareHouse_sl = spark.emptyDataset[Record].toDF
    val wareHouse_sl_sizeLimit = 10000
    
    val ssc = new StreamingContext(spark.sparkContext, Seconds(BATCH_INTERVAL))
   
    val inputDataStream = ssc.socketTextStream("localhost", 2222, StorageLevel.MEMORY_AND_DISK_SER)
     
    // streaming job going to go now
    inputDataStream.foreachRDD(rdd => {
      
        import spark.implicits._  
        val recordDF = rdd.map { textString => TempRecord( Utilities.parse(textString, "timestamp"), Utilities.parse(textString, "title"),Utilities.parse(textString, "content")) }.toDF
        
        val tokenizer = new Tokenizer()
		    .setInputCol("content")
		    .setOutputCol("contentToWords")
		    .transform(recordDF)
		    
		    //Remove the content column
		    val mainDF = tokenizer.select("timestamp", "title", "contentToWords")
		    
		    //Collect Records for training initial model
		    if(initialModelFlag){
		      
		      firstModelDF = firstModelDF.union(mainDF)
		      
		      if( firstModelDF.count() > initialTrainingRecords ){
		        initialModelFlag = false
		        
		        //place firstModelDF into wareHouse_fl
		        wareHouse_fl = wareHouse_fl.union(firstModelDF)
		        
		        println("Models are being generated ......")
		        
		        //Train initial model
		        computeModelHierarchy(spark, firstModelDF, firstLevelClustersNum, firstLevelClusterMaxItr, secondLevelClustersNum, secondLevelClusterMaxItr)
		        
		        println("Model generated successfully...")
		        
		      }  
		    }
        
        //Testing 
		    else{
		      
		      //Insert this RDD into wareHouse_fl
		      wareHouse_fl = mainDF.union(wareHouse_fl)
		      
		      //when the size of warehouse excides its limit then take first 15000 articels and discard the olders one
		      if(wareHouse_fl.count() > wareHouse_fl_sizeLimit){
		          wareHouse_fl = wareHouse_fl.limit(wareHouse_fl_sizeLimit)
		      }
		         		      
		      //Load df model
		      val firstLevelTFModel = CountVectorizerModel.load("models/featuresSpace/TFfirstLevelModel")
		      //Load the tfidf model
		      val firstLevelTFIDFModel = IDFModel.load("models/featuresSpace/TFIDFfirstLevelModel")
		      //Load first level kmeans model
		      val firstLeveTopiclModel = KMeansModel.load("models/firstLeveTopiclModel")
		      
		      //transform dataset
		      val fltf = firstLevelTFModel.transform(mainDF)
		      val fltfidf = firstLevelTFIDFModel.transform(fltf).filter { row => row.getAs[SparseVector]("featuresTFIDF").values.size != 0 }
		      val flTransDF = firstLeveTopiclModel.transform(fltfidf)
		      println("---------------------Here goes the final transformations---------------------")
		      flTransDF.show()
		      
		      //cluster centers
		      val flClustersCernterVector = firstLeveTopiclModel.clusterCenters
		      firstLevelClustersCrenter = flClustersCernterVector.map { _.toArray }
		  
		      //Iterate over all the first level clusters
		      for (flClusterIndex <- 0 until firstLevelClustersNum){
		        
		        val eachFLClusterDF = flTransDF.filter { row => row.getAs[Int]("clusterPrediction") == flClusterIndex }
		        
		        //add a new column named 'distance' that will have distance between cluster center and data points
		        val withDistance = eachFLClusterDF.withColumn("distance", computeSQD(eachFLClusterDF.col("clusterPrediction"), eachFLClusterDF.col("featuresTFIDF")))
		        
		        /*
		         * Case 1: documents those do not fit into any of the first level topics, insert them into first level container
		         */
		        val newDocs = withDistance.filter {row => row.getAs[Double]("distance") > firstLevelThreshold}
		        container_fl = container_fl.union(newDocs.select("timestamp", "title", "contentToWords"))
		        if (container_fl.count() >= container_fl_sizeLimit){
		          //insert it into first Level Warehouse
		          wareHouse_fl = container_fl.union(wareHouse_fl)
		          //Compute the first level and second level model
		          computeModelHierarchy(spark, wareHouse_fl, firstLevelClustersNum, firstLevelClusterMaxItr, secondLevelClustersNum, secondLevelClusterMaxItr)
		        }
		        
		        /*
		         * Case 2: documents those fits into first level cluster
		         * then
		         * 	-find sub-cluster its fit into
		         * 	-compute distance between document data points and its sub-cluster center
		         * 	-place the documents either in second level container or second level warehouse
		         */
		        else{
		          
		          //documents those fits into first level topics
		          val flFitDocs = withDistance.filter(row => row.getAs[Double]("distance") <= firstLevelThreshold)
		         
		          //extract only the row columns
		          val rowFLFitDocs = flFitDocs.select("timestamp", "title", "contentToWords")
		          
		          //load the second level tf, tfidf and k-means model
		          //we are at the place where we have all the docs are belong to the 'first' first level topics 
		          
		          
		        }
		        
		        //val oldDocs = withDistance.filter { row => row.getAs[Double]("distance") <= firstLevelThreshold }
		        
		      }//end of for loop
		      
		    }//end of main-else   
      }
    )
       
    ssc.start()
    ssc.awaitTermination()
    
  }//end of main
  
  /**
   * User defined functions
   * 
   * to compute squire distance
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
  
  //Compute feature space
  def computeFeatureSpace(preProcessedData: Dataset[Row], vocabSize: Int, minDF: Int, featuresModelName: String): Dataset[Row] = {
    
    //Term-frequencies vector
	 val tfModel = new CountVectorizer()
		.setInputCol("contentToWords")
		.setOutputCol("featuresTF")
		.setVocabSize(vocabSize)
		.setMinDF(minDF)
		.fit(preProcessedData)
		
		//firstLevelTF
		tfModel.write.overwrite().save(s"models/featuresSpace/TF$featuresModelName")
		
		val featurizedData  = tfModel.transform(preProcessedData)
	  //featurizedData.show()
	  
	  //TF-IDF vector
		val tfidfModel = new IDF()
		  .setInputCol("featuresTF")
		  .setOutputCol("featuresTFIDF")  //fist level topics features space
		  .fit(featurizedData)
		  
	 tfidfModel.write.overwrite().save(s"models/featuresSpace/TFIDF$featuresModelName")
		  
	 val tfidf = tfidfModel.transform(featurizedData)
  
	 return tfidf
	     
  }
  
  //This method computes topic from text documents collection
  //Input column "contentToWords", should be already pre-processed
  def computeTopic(dataset: Dataset[Row]): String = {
    
   var topic: String = ""
    
    //Term-frequencies vector
	 val tfModel = new CountVectorizer()
		.setInputCol("contentToWords")
		.setOutputCol("featuresTF")
		.setVocabSize(20)
		.fit(dataset)
		
	 val tf  = tfModel.transform(dataset)
	
  //TF-IDF vector
	val tfidfModel = new IDF()
	  .setInputCol("featuresTF")
	  .setOutputCol("featuresTFIDF")
	  .fit(tf)
	  
	val vocab = tfModel.vocabulary
	val tfidfWeight  = tfidfModel.idf.toArray
	
	val vocabAndWeight = vocab.map { term => (term, tfidfWeight(vocab.indexOf(term))) }
  
	//now sort by weight
	val sortedVocabAndWeight = vocabAndWeight.sortWith((tuple1, tuple2) => tuple1._2 > tuple2._2)
	
	//sortedVocabAndWeight.foreach(println)
	 
	val impoTopics = sortedVocabAndWeight.map((tuple) => tuple._1)
	
	//argument to take (5) is the number of vocabularies terms used for topic
	impoTopics.take(4).foreach { term => topic = topic + " "+term }
	
	return topic
		
  }
  
  //computes firstLevel and secondLevel k-means model and store them in dir 'models'
  //Input data must have column named 'features'
  def computeModelHierarchy(spark: SparkSession, preProcessedDataset: Dataset[Row], firstLevelClustersNum: Int, firstLevelClustersMaxItr: Int, secondLevelClustersNum: Int, secondLevelClustersMaxItr: Int): Unit = {
    
    //Compute features space for first level topics
		val tfidfForFLT = computeFeatureSpace(preProcessedDataset, firstLevelVocalSize, firstLevelMinDF, "firstLevelModel")
    
		//cache data as K-means will make multiple iteration over data set
    tfidfForFLT.cache()
    
    //First level model
		val firstLevelKmeansModel = new KMeans()
		  .setK(firstLevelClustersNum)
		  .setInitMode("k-means||")
		  .setMaxIter(firstLevelClustersMaxItr)
		  .setFeaturesCol("featuresTFIDF")
		  .setPredictionCol("clusterPrediction")
		  .fit(tfidfForFLT)
		
	  //Save model on the file
		firstLevelKmeansModel.write.overwrite().save("models/firstLeveTopiclModel")
		
		//Load model from the file 
		//val kmeansModel = KMeansModel.load("models/firstLevelModel")
		
		val firstLevelKmeansModelSummary = firstLevelKmeansModel.summary
		
		//DataFrame with prediction column
		val firstLevelDF = firstLevelKmeansModelSummary.predictions
		//firstLevelDF.show(40)
		
		//save as temporary table to run SQL queries
		firstLevelDF.createOrReplaceTempView("firstLevelTable")
		
		println("All clusters are: ")
		//Iterate over all the clusters
		for (clusterIndex <- 0 until firstLevelClustersNum ){
		  
		  val clusterDF = spark.sql(s"SELECT * FROM firstLevelTable WHERE clusterPrediction = $clusterIndex")
		  
		  //Drop the following columns as they are not needed for computing model for second level topics: featuresTF, featuresTFIDF and clusterPrediction
		  val clusterDFWithRemovedFS = clusterDF.drop("featuresTF", "featuresTFIDF", "clusterPrediction")  //DataSet with removed features space
		  //clusterDFWithRemovedFS.show()
		  
		  //compute features space for second level topic
		  val tfidfForSLT = computeFeatureSpace(clusterDFWithRemovedFS, secondLevelVocalSize, secondLevelMinDF, s"subModel_$clusterIndex")
		  
		  tfidfForSLT.cache()
		  
		  //Prepare model for second level topics
		  val secondLevelKmeansModel = new KMeans()
		    .setK(secondLevelClustersNum)
		    .setInitMode("k-means||")
		    .setMaxIter(secondLevelClustersMaxItr)
		    .setFeaturesCol("featuresTFIDF")
		    .setPredictionCol("clusterPrediction")
		    .fit(tfidfForSLT)
		    
	    //save model on file
		  secondLevelKmeansModel.write.overwrite().save(s"models/secondLeveTopiclModel/subModel_$clusterIndex")
		  
		  //Create a temporary table for each set of subtopics
		  secondLevelKmeansModel.transform(tfidfForSLT).createOrReplaceTempView(s"secondLevelTable_$clusterIndex")
		  
		  //secondLevelKmeansModel.transform(tfidfForSLT).sort("clusterPrediction").show()
		 
		}		
  }
  
  //Computes topics Hierarchy
  //This method assume that temp
  def computeTopicsHierarchy(spark: SparkSession): Dataset[Row] = {
    
    //Holds final topics tuple(mainTopic, subTopic)
    var topicsSet: Set[(String, String)] = Set()
    
    for (firstLevelClusterIndex <- 0 until firstLevelClustersNum){
      
      val firstLevelClusterDF = spark.sql(s"SELECT contentToWords FROM firstLevelTable WHERE clusterPrediction = $firstLevelClusterIndex")
      val mainTopic = computeTopic(firstLevelClusterDF)
      
      //Iterate over all its sub-clusters
      for(secondLevelClusterIndex <- 0 until secondLevelClustersNum){
        
        val secondLevelClusterDF = spark.sql(s"SELECT contentToWords FROM secondLevelTable_$firstLevelClusterIndex WHERE clusterPrediction = $secondLevelClusterIndex")
        val subTopic = computeTopic(secondLevelClusterDF)
        
        topicsSet = topicsSet+((mainTopic, subTopic))
      } 
    }
    
    println("List of First level topics are :-")
    topicsSet.foreach(println)
     
    val topicsDF = spark.createDataFrame(topicsSet.toSeq).toDF("MainTopic", "SubTopic")
    val sortedTopicsDF = topicsDF.sort("MainTopic")
    
    //save this DF as a temp table
    sortedTopicsDF.createOrReplaceTempView("topicsTable")
    
    println("Topic DataFrame: ")
    sortedTopicsDF.show(false)
   
    return topicsDF
  }
  
  //Will be used for evaluation
  //Computes cost associated with each cluster (firstLevel or secondLevel)
  //Call this methods after model has been created (after computeModelHierarchy(...))
  def computeClustersCost(spark: SparkSession): Unit ={
    
    case class ClusterCost(timeStamp: String, flClusterIndex: Int, flClusterCost: Double, slClusterIndex: Int, slClusterCost: Double )
    
    val costsArray = new Array[ClusterCost](firstLevelClustersNum * secondLevelClustersNum)
    var costsArrayIndex = 0
    
    //spark.sparkContext.parallelize(seq, numSlices)
    
    //Load the first level clusters model
    val firstLevelKMeansModel = KMeansModel.load("models/firstLeveTopiclModel")
    
    //Iterate over first level clusters
    for (firstLevelClusterIndex <- 0 until firstLevelClustersNum){
      
      val firstLevelClusterDF = spark.sql(s"SELECT featuresTFIDF FROM firstLevelTable WHERE clusterPrediction = $firstLevelClusterIndex")
      
      val firstLevelClusterCost: Double = firstLevelKMeansModel.computeCost(firstLevelClusterDF)
      
      //Load the corresponding second level model
      val secondLevelKMeansModel = KMeansModel.load(s"models/secondLeveTopiclModel/subModel_$firstLevelClusterIndex")
      
     
      //Iterate over all the second level clusters
      for (secondLevelClusterIndex <- 0 until secondLevelClustersNum){
        
        val secondLevelClusterDF = spark.sql(s"SELECT featuresTFIDF FROM secondLevelTable_$firstLevelClusterIndex WHERE clusterPrediction = $secondLevelClusterIndex")  
        
        val secondLevelClusterCost = secondLevelKMeansModel.computeCost(secondLevelClusterDF)
        
        costsArray(costsArrayIndex) = ClusterCost("TimeStamp", firstLevelClusterIndex, firstLevelClusterCost, secondLevelClusterIndex, secondLevelClusterCost)
        costsArrayIndex = costsArrayIndex + 1      
      }    
    }
    
    val costsArrayRDD = spark.sparkContext.parallelize(costsArray)
      .map { ca => Row(ca.timeStamp, ca.flClusterIndex, ca.flClusterCost, ca.slClusterIndex, ca.slClusterCost) }
    
    //convert to DataFrame
    val schema = StructType(Array[StructField](
          StructField("TimeStamp", StringType, nullable = true),
          StructField("firstLevelClusterIndex", IntegerType, nullable = true),
          StructField("firstLevelClusterCost", DoubleType, nullable = true),
          StructField("secondLevelClusterIndex", IntegerType, nullable = true),
          StructField("secondLevelClusterCost", DoubleType, nullable = true)
        ))
    
    val costsArrayDF = spark.createDataFrame(costsArrayRDD, schema)
    
    costsArrayDF.show(false)  
    
  }
    
}//end of object


/** Case class for converting RDD to DataFrame ("timeStamp fileName content")*/
case class TempRecord(timestamp: String, title: String, content: String)

/** Case class for DF ("timeStamp fileName contentToWords ") */
case class Record(timestamp: String, title: String, contentToWords: Array[String] )

/**
 * This object provides with helper API
 */
object Utilities {
  
  
  /**
   * This method parse the input 'text' and return the content of 'tag'
   * @input: 'text' format- <timestamp>123456</timestamp><title>title goes here</title><content>content goes here</content>
   * 			 : 'tag' can be one of the three - timestamp, title, content
   * 
   * @output: content of input tag 
   */
  def parse(text: String, tag: String): String = {
    
    var result = "";
    
    if(tag.equalsIgnoreCase("timestamp")){  
      val timestampPattern = """<timestamp>.*</timestamp>""".r
      val timestamp = timestampPattern.findFirstIn(text)
      if(timestamp.isDefined)
        result = timestamp.get.replace("<timestamp>", "").replace("</timestamp>", "") 
    }
    
    else if(tag.equalsIgnoreCase("title")){
      val titlePattern = """<title>.*</title>""".r
      val title = titlePattern.findFirstIn(text)
      if(title.isDefined)
        result = title.get.replace("<title>", "").replace("</title>", "").replaceAll("""\s+""", " ").trim      
    }
    
    else if(tag.equalsIgnoreCase("content")){
      val contentPattern = """<content>.*</content>""".r
      val content = contentPattern.findFirstIn(text)
      if(content.isDefined)
        result = content.get.replace("<content>", "").replace("</content>", "").trim
    }
    else {
      result = ""
    }
    
    result;
    
  }// end of parser()
 
}//end of Utilities

/*
 * FeaturesSpace model
 * First Level: models/featuresSpace/TF+firstLevelModel, TFIDF+firstLevelModel
 * Second level: models/featuresSpace/TF+subModel_clusterIndex, TFIDF+subModel_clusterIndex
 * 
 */



















