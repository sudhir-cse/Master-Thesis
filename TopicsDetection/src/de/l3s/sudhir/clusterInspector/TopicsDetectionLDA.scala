package de.kbs.thesis

import org.apache.spark.ml.clustering.KMeansSummary
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.clustering.LDAModel
import scala.collection.mutable.WrappedArray


object TopicsDetectionLDA {

	    val NUMBER_OF_FIRST_LEVEL_TOPICS = 3     //Number of clusters, value of k
			val FIRST_LEVEL_TOPIS_LENGTH = 3      //In words
			val FIRST_LEVEL_MAX_ITR = 20
			val FIRST_LEVEL_VOC_SIZE = 20
			val FIRST_LEVEL_MIN_DF = 0
			val FIRST_LEVEL_CLUSTER_MIN_PROBABILITY: Double = 0.0 //Documents belong to the topics that has distribution grater than 0.6

			val NUMBER_OF_SECOND_LEVEL_TOPICS = 2
			val SECOND_LEVEL_TOPICS_LENGTH = 3
			val SECOND_LEVEL_MAX_ITR = 20
			val SECOND_LEVEL_VOC_SIZE = 10
			val SECOND_LEVEL_MIN_DF = 0
			val SECOND_LEVEL_CLUSTER_MIN_PROBABILITY: Double = 0.0

			//To store set of documents that belong to each topic
			val TOPIC_DOCUMENTS_DFs: Array[Dataset[Row]] = new Array[Dataset[Row]](NUMBER_OF_FIRST_LEVEL_TOPICS)
			
			//To store first level topics
			var FIRST_LEVEL_TOPIC_DF: Dataset[Row] = null
			
			//To store second level topics
			val SECONG_LEVEL_TOPICS_DFs: Array[Dataset[Row]] = new Array[Dataset[Row]](NUMBER_OF_FIRST_LEVEL_TOPICS)

			//Main function
			def main(args: Array[String]): Unit = {

		//Create spark session with the name 'Clustering in Archive Collection' that runs locally on all the core 
		val spark = SparkSession.builder()
				.master("local[*]")
				.appName("Clustering in Archeive Collection")
				.config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse")
				.getOrCreate()

				//RDD[Row(timeStemp, fileName, fileContent)]
				//File name is only the time stamp stamp
				val trainingData = spark.sparkContext.wholeTextFiles("data/KDDTraining")

				//Pre-process Dataset
				val preProcessedTrainingData = preProcessData(spark, trainingData)

				//Computes LDA models hierarchy
				computeLDAModelsAndTopicsHierarchy(spark, preProcessedTrainingData)

				//Prints topics hierarchy
				printTopicsHierarchy(spark)

	}

	//Pre-processing data: RDD[(String, String)]
	def preProcessData(spark: SparkSession, data: RDD[(String, String)]): Dataset[Row] = {

		//RDD[Row(timeStemp, fileName, fileContent)]
		//Filename has been composed of Timestamp and Filename. Separator as a "-" has been used
		//Data filterring includes: toLoweCasae, replace all the white space characters with single char, keep only alphabetic chars, keep only the words > 2.
		val tempData = data.map(kvTouple => Row(
				//DateTimeFormat.forPattern("YYYY-MM-dd").print(kvTouple._1.split(".")(0).toLong),
				kvTouple._1,
				"FileName", 
				kvTouple._2.toLowerCase().replaceAll("""\s+""", " ").replaceAll("""[^a-zA-Z\s]""", "").replaceAll("""\b\p{IsLetter}{1,2}\b""","")
				))

				//Convert training RDD to DataFrame(timestamp, fileName, fileContent)
				//Schema is encoded in String
				val schemaString = "timeStamp fileName fileContent"

				//Generate schema based on the string of schema
				val fields = schemaString.split(" ")
				.map ( fieldName => StructField(fieldName, StringType, nullable = true) )

				//Now create schema
				val schema = StructType(fields)

				//Apply schema to RDD
				val trainingDF = spark.createDataFrame(tempData, schema)
				//trainingDF.show()

				//split fileContent column into words
				val wordsData = new Tokenizer()
		.setInputCol("fileContent")
		.setOutputCol("words")
		.transform(trainingDF)

		//Remove stop words
		val stopWordsRemoved = new StopWordsRemover()
		.setInputCol("words")
		.setOutputCol("stopWordsFiltered")
		//.setStopWords(StopWordsRemover.loadDefaultStopWords("english"))
		.transform(wordsData)

		return stopWordsRemoved
	}

	//Computes LDA Models and topics hierarchy 
	def computeLDAModelsAndTopicsHierarchy(spark: SparkSession, preProcessedDataset: Dataset[Row]): Unit = {

		//Term-frequencies vector
		//LDA requires only TF as input
		val tfModelF = new CountVectorizer()
		.setInputCol("stopWordsFiltered")
		.setOutputCol("featuresTFF")
		.setVocabSize(FIRST_LEVEL_VOC_SIZE)
		.setMinDF(FIRST_LEVEL_MIN_DF)
		.fit(preProcessedDataset)

		//save model
		tfModelF.write.overwrite().save("LDAModels/featuresSpace/firstLevelTF")

		val featurizedDataF  = tfModelF.transform(preProcessedDataset)

		featurizedDataF.cache()

		//Train LDA Model
		val ldaModelF = new LDA()
		.setK(NUMBER_OF_FIRST_LEVEL_TOPICS)
		.setMaxIter(FIRST_LEVEL_MAX_ITR)
		.setFeaturesCol("featuresTFF")
		.setTopicDistributionCol("topicDistributionF")
		.fit(featurizedDataF)

		//save LDA model
		ldaModelF.write.overwrite().save("LDAModels/firstLeveTopiclModel")

		//Save as temp view
		ldaModelF.transform(featurizedDataF).createOrReplaceTempView("firstLevelTopicsView")

		//compute the first level topics DataFrame
		val firstLevelVocalsF = tfModelF.vocabulary

		val describeTopiceDFF = ldaModelF.describeTopics(FIRST_LEVEL_TOPIS_LENGTH)

		//User defined function to compute topics name
		val udfIndicesToTopicsF = udf[Array[String], WrappedArray[Int]](indices => {

			indices.toArray.map {index => firstLevelVocalsF(index)}

		})

		//use user defined functions
		val topicsDFF = describeTopiceDFF.withColumn("firstLevelTopic", udfIndicesToTopicsF(describeTopiceDFF.col("termIndices")))

		//topicsDFF.show()

		//save as a table
		topicsDFF.createOrReplaceTempView("firstLevelTopicsView")
		
		FIRST_LEVEL_TOPIC_DF = topicsDFF
		

		/*--------------Work towards second level models and topics----------------*/

		//Filter out each topic and documents in it and store them in topicsDFs array
		val transDataSet = ldaModelF.transform(featurizedDataF)

		for (topicIndex <- 0 to NUMBER_OF_FIRST_LEVEL_TOPICS-1 ){

			val firstTopicsDocumentsSet = transDataSet.filter { row => row.getAs[DenseVector]("topicDistributionF").toArray(topicIndex) >= FIRST_LEVEL_CLUSTER_MIN_PROBABILITY }

			TOPIC_DOCUMENTS_DFs(topicIndex) = firstTopicsDocumentsSet

		} 

		/*---------- Compute second level topics model ----------*/
		TOPIC_DOCUMENTS_DFs.indices.foreach { i => {

			//Term-frequencies vector
			//LDA requires only TF as input
			val tfModelS = new CountVectorizer()
			.setInputCol("stopWordsFiltered")
			.setOutputCol("featuresTFS")
			.setVocabSize(SECOND_LEVEL_VOC_SIZE)
			.setMinDF(SECOND_LEVEL_MIN_DF)
			.fit(TOPIC_DOCUMENTS_DFs(i))

			//save model
			tfModelS.write.overwrite().save("LDAModels/featuresSpace/secondLevelTF"+i)

			val featurizedDataS  = tfModelS.transform(TOPIC_DOCUMENTS_DFs(i))

			featurizedDataS.cache()

			//Train LDA Model
			val ldaModelS = new LDA()
			.setK(NUMBER_OF_SECOND_LEVEL_TOPICS)
			.setMaxIter(SECOND_LEVEL_MAX_ITR)
			.setFeaturesCol("featuresTFS")
			.setTopicDistributionCol("topicDistributionS")
			.fit(featurizedDataS)

			//save LDA model
			ldaModelS.write.overwrite().save("LDAModels/secondLeveTopicsModel"+i)

			//save the transformation as a temp view
			ldaModelS.transform(featurizedDataS).createOrReplaceTempView("secondLevelTopicsView"+i)

			//compute the second level topics DataFrame
			val secondLevelVocalsS = tfModelS.vocabulary

			val describeTopiceDFS = ldaModelS.describeTopics(SECOND_LEVEL_TOPICS_LENGTH)

			//User defined function to compute topics name
			val udfIndicesToTopicsS = udf[Array[String], WrappedArray[Int]](indices => {

				indices.toArray.map {index => secondLevelVocalsS(index)}

			})

			//use user defined functions
			val topicsDFS = describeTopiceDFS.withColumn("secondLevelTopic", udfIndicesToTopicsS(describeTopiceDFS.col("termIndices")))

			//topicsDFF.show()

			//save as a table
			topicsDFS.createOrReplaceTempView("secondLevelTopicsView"+i)
			
			SECONG_LEVEL_TOPICS_DFs(i) = topicsDFS
			
		} 
		}	
	}

	//Prints topics hierarchy
	def printTopicsHierarchy(spark: SparkSession): Unit = {
	  
	  println("First and second level topics are as follows : ")
	  
	  val firstLevelTopic = FIRST_LEVEL_TOPIC_DF.collect
	  val secondLevelTopicsArray = SECONG_LEVEL_TOPICS_DFs.map { slTopicDS => slTopicDS.collect }
	  
	  firstLevelTopic.foreach { flTopicRow => {
	    
	    val flTopic = flTopicRow.getAs[WrappedArray[String]]("firstLevelTopic")
	    val flTopicIndex = flTopicRow.getAs[Integer]("topic")
	    
	    flTopic.toArray.foreach{ temp => print(temp+" ")}
	    print("\n")
	    print("\t")
	    
	    secondLevelTopicsArray(flTopicIndex).foreach { slTopicRow => {
	      
	      val slTopic = slTopicRow.getAs[WrappedArray[String]]("secondLevelTopic")
	      
	      slTopic.toArray.foreach { temp => print(temp+" ")}
	      
	      print("\n")
	      print("\t")
	      
	      
	    } }
	    
	    println("\n----------------")
	    
	  } }
	  
		
	  
//		spark.sql("SELECT firstLevelTopic FROM firstLevelTopicsView").foreach { rowF => {
//		  
//		  val flTopic = rowF.getAs[Array[String]]("firstLevelTopic").toString()
//		  println(flTopic)
//
//		} }
//	  
//	  println("Second level topics DFs are:")
//	  for (flClusterIndex <- 0 until NUMBER_OF_FIRST_LEVEL_TOPICS ){
//	    
//	    spark.sql(s"SELECT secondLevelTopic FROM secondLevelTopicsView$flClusterIndex").show()
//	    
//	  }
	  
	  
	  
	}
}

//next questions
//Understand the DataSet, Find the way to run this algorithm on top of them, either download them locally or run on the server itself
//Exception handling -> java.lang.IllegalArgumentException: requirement failed: The vocabulary size should be > 0. Lower minDF as necessary