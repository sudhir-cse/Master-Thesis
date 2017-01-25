package de.l3s.sudhir.clusterInspector

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
import scala.io.StdIn.{readLine,readInt}
import de.l3s.sudhir.clusterInspector

/**
 * Outputs {-word2vec (300 dimensions) -> PCA (2 dimension) vector and labels} for the following
 * 	-for entire corpus
 * 	-for each first level cluster
 * 	-for each second level cluster
 * 	
 * 	-a single file containing only first level vocabularies
 * 	-another single file containing only second level vocabularies
 * 	
 */

object KMeansInspector {
  
  var INPUT_FILE:String = ""
  var OUTPUT_DIR:String = ""
  
  //KMeans setting parameters
  var firstLevelClustersNum: Int = 3
  var firstLevelClusterMaxItr: Int = 100
  var firstLevelVocalSize: Int = 100
  var firstLevelMinDF: Int = 5

  var secondLevelClustersNum: Int = 2
  var secondLevelClusterMaxItr: Int = 100
  var secondLevelVocalSize: Int = 50
  var secondLevelMinDF: Int = 1
  
  var MIN_PARTITION: Int = 35
  
  val sparkWarehouse = "temp/spark-warehouse";
  
  var firstLevelClustersCrenter: Array[Array[Double]] = null
  
  /** ---------------- main() ----------------  */
	  
  def main(args: Array[String]): Unit = {
   
    //Read input and output files from terminal
    println("---------Input file and Output directory-------")
    INPUT_FILE = readLine("Please Provide the Input FILE: ")
    OUTPUT_DIR = readLine("Please Provide the Output Directory: ")
    print("\nPlease Enter Minimum Number of Partition: ")
    MIN_PARTITION = readInt()
    print("\n\n")
    
    println("------------First Level Cluster-------------")
    print("Please Enter the values for:\n")
    print("\tK:  ")
    firstLevelClustersNum = readInt()
    print("\n\tMax Iteration:  ")
    firstLevelClusterMaxItr = readInt()
    print("\n\tVocabularies Size:  ")
    firstLevelVocalSize = readInt()
    print("\n\tMin Document Frequency:  ")
    firstLevelMinDF = readInt()
    print("\n\n")
    
    println("------------Second Level Cluster-------------")
    print("Please Enter the values for:\n")
    print("\tK:  ")
    secondLevelClustersNum = readInt()
    print("\n\tMax Iteration:  ")
    secondLevelClusterMaxItr = readInt()
    print("\n\tVocabularies Size:  ")
    secondLevelVocalSize = readInt()
    print("\n\tMin Document Frequency:  ")
    secondLevelMinDF = readInt()
    
    println("\nDone. Thanks!")
    
    val spark = SparkSession.builder()
    .master("local[*]")
    .appName("KMeans Clusters Inspector")
    .config("spark.sql.warehouse.dir", sparkWarehouse)
    .getOrCreate()
    
    val inputData = spark.sparkContext.textFile(INPUT_FILE, MIN_PARTITION)
  
    import spark.implicits._  
    val recordDF = inputData.map { textString => TempRecord( Utilities_Inspector.parse(textString, "timestamp"), Utilities_Inspector.parse(textString, "title"),Utilities_Inspector.parse(textString, "content")) }.toDF
    
    val tokenizer = new Tokenizer()
		    .setInputCol("content")
		    .setOutputCol("contentToWords")
		    .transform(recordDF)
		    
		//Remove the content column
    val mainDF = tokenizer.select("timestamp", "title", "contentToWords")
    
    //compute word2vec for all collection
    val w2v = word2vec.computeWord2Vec(mainDF.select("contentToWords"))
    val reduceTo2Dim = pca.reduceDimensionTo2(w2v) //DataFrame: word, vector, pcaFeatures
    
    //write columns word and pcaFeatures in different files
    reduceTo2Dim.select("word").rdd.repartition(1).saveAsTextFile(OUTPUT_DIR+"/firstLevel/flLables")
    reduceTo2Dim.select("pcaFeatures").rdd.repartition(1).saveAsTextFile(OUTPUT_DIR+"/firstLevel/flVectors")
    
    //Compute model hierarchy
    computeModelHierarchy(spark, mainDF, firstLevelClustersNum, firstLevelClusterMaxItr, secondLevelClustersNum, secondLevelClusterMaxItr)
    
    recordDF
  
    spark.stop()
    
  }//end of main
  
  
  //Compute feature space as well prints the vocabularies
  def computeFeatureSpace(spark:SparkSession, preProcessedData: Dataset[Row], vocabSize: Int, minDF: Int, vocal: String): Dataset[Row] = {
    
    //Term-frequencies vector
	 val tfModel = new CountVectorizer()
		.setInputCol("contentToWords")
		.setOutputCol("featuresTF")
		.setVocabSize(vocabSize)
		.setMinDF(minDF)
		.fit(preProcessedData)
		
	  val vocabularies = tfModel.vocabulary
	  val vocalDF = spark.createDataFrame(vocabularies.map(Tuple1.apply)).toDF("vocal")
	  vocalDF.rdd.repartition(1).saveAsTextFile(OUTPUT_DIR+"/vocabularies/"+vocal)
	  
		
		val featurizedData  = tfModel.transform(preProcessedData)
	  //featurizedData.show()
	  
	  //TF-IDF vector
		val tfidfModel = new IDF()
		  .setInputCol("featuresTF")
		  .setOutputCol("featuresTFIDF")  //fist level topics features space
		  .fit(featurizedData)
		  
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
		val tfidfForFLT = computeFeatureSpace(spark,preProcessedDataset, firstLevelVocalSize, firstLevelMinDF, "flVocal")
    
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
		
		val firstLevelKmeansModelSummary = firstLevelKmeansModel.summary
		
		//DataFrame with prediction column
		val firstLevelDF = firstLevelKmeansModelSummary.predictions
		
		//save as temporary table to run SQL queries
		firstLevelDF.createOrReplaceTempView("firstLevelTable")
		
		
		
		println("All clusters are: ")
		//Iterate over all the clusters
		for (clusterIndex <- 0 until firstLevelClustersNum ){
		  
		  val clusterDF = spark.sql(s"SELECT * FROM firstLevelTable WHERE clusterPrediction = $clusterIndex")
		  
		  //Drop the following columns as they are not needed for computing model for second level topics: featuresTF, featuresTFIDF and clusterPrediction
		  val clusterDFWithRemovedFS = clusterDF.drop("featuresTF", "featuresTFIDF", "clusterPrediction")  //DataSet with removed features space
		  //clusterDFWithRemovedFS.show()
		  
		  //word2vec and pca contentToWords
      val w2vSL = word2vec.computeWord2Vec(clusterDFWithRemovedFS.select("contentToWords"))
      val reduceTo2DimSL = pca.reduceDimensionTo2(w2vSL) //DataFrame: word, vector, pcaFeatures
      
      //write columns word and pcaFeatures in different files
      reduceTo2DimSL.select("word").rdd.repartition(1).saveAsTextFile(OUTPUT_DIR+"/secondLevel/slLables-"+clusterIndex)
      reduceTo2DimSL.select("pcaFeatures").rdd.repartition(1).saveAsTextFile(OUTPUT_DIR+"/secondLevel/slVectors-"+clusterIndex)
		  
		  //compute features space for second level topic
		  val tfidfForSLT = computeFeatureSpace(spark, clusterDFWithRemovedFS, secondLevelVocalSize, secondLevelMinDF, s"subVocal_$clusterIndex")
		  
		  tfidfForSLT.cache()
		  
		  //Prepare model for second level topics
		  val secondLevelKmeansModel = new KMeans()
		    .setK(secondLevelClustersNum)
		    .setInitMode("k-means||")
		    .setMaxIter(secondLevelClustersMaxItr)
		    .setFeaturesCol("featuresTFIDF")
		    .setPredictionCol("clusterPrediction")
		    .fit(tfidfForSLT)
		  
		  //Create a temporary table for each set of subtopics
		  secondLevelKmeansModel.transform(tfidfForSLT).createOrReplaceTempView(s"secondLevelTable_$clusterIndex")
		  
		  //iterate over subclusters of each second level cluster
		  for (ci <- 0 until secondLevelClustersNum ){
		    
		    val clusterDFSSL = spark.sql(s"SELECT * FROM secondLevelTable_$clusterIndex WHERE clusterPrediction = $ci")
		    
		    //word2vec and pca contentToWords
        val w2vSSL = word2vec.computeWord2Vec(clusterDFSSL.select("contentToWords"))
        val reduceTo2DimSSL = pca.reduceDimensionTo2(w2vSSL) //DataFrame: word, vector, pcaFeatures
        
        //write columns word and pcaFeatures in different files
        reduceTo2DimSSL.select("word").rdd.repartition(1).saveAsTextFile(OUTPUT_DIR+"/secondSecondLevel/sslLables-"+clusterIndex)
        reduceTo2DimSSL.select("pcaFeatures").rdd.repartition(1).saveAsTextFile(OUTPUT_DIR+"/secondSecondLevel/sslVectors-"+clusterIndex)
		    
		  }  
		 
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
      
}//end of object


/** Case class for converting RDD to DataFrame ("timeStamp fileName content")*/
case class TempRecord(timestamp: String, title: String, content: String)

/** Case class for DF ("timeStamp fileName contentToWords ") */
case class Record(timestamp: String, title: String, contentToWords: Array[String] )




















