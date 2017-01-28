package de.l3s.sudhir.LDA

import org.apache.spark.ml.clustering.KMeansSummary
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.clustering.LDAModel
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.collection.mutable.WrappedArray
import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer

import java.util.List
import java.util.ArrayList
import java.util.Properties

import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.ling.CoreAnnotations._



object TopicsDetectionLDA {

	    val NUMBER_OF_FIRST_LEVEL_TOPICS = 4     //Number of clusters, value of k
			val FIRST_LEVEL_TOPIS_LENGTH = 4      //In words
			val FIRST_LEVEL_MAX_ITR = 20
			val FIRST_LEVEL_VOC_SIZE = 60
			val FIRST_LEVEL_MIN_DF = 15
			val FIRST_LEVEL_CLUSTER_MIN_PROBABILITY: Double = 0.3 //Documents belong to the topics that has distribution grater than 0.6

			val NUMBER_OF_SECOND_LEVEL_TOPICS = 3
			val SECOND_LEVEL_TOPICS_LENGTH = 3
			val SECOND_LEVEL_MAX_ITR = 15
			val SECOND_LEVEL_VOC_SIZE = 40
			val SECOND_LEVEL_MIN_DF = 10
			val SECOND_LEVEL_CLUSTER_MIN_PROBABILITY: Double = 0.1

			//To store set of documents that belong to each topic
			val TOPIC_DOCUMENTS_DFs: Array[Dataset[Row]] = new Array[Dataset[Row]](NUMBER_OF_FIRST_LEVEL_TOPICS)
			
			//To store first level topics
			var FIRST_LEVEL_TOPIC_DF: Dataset[Row] = null
			
			//To store second level topics
			val SECONG_LEVEL_TOPICS_DFs: Array[Dataset[Row]] = new Array[Dataset[Row]](NUMBER_OF_FIRST_LEVEL_TOPICS)
			
			/*
			 * English stop words list
			 * Reference:  http://xpo6.com/list-of-english-stop-words/
			 */
			val ENGLISH_STOP_WORDS_LIST  = Set("a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the");

			/*-----------------------------------------------------------------------------------------------------------*/
	         /*------------------------------------    main()  -------------------------------------------------*/
			/*-----------------------------------------------------------------------------------------------------------*/
			def main(args: Array[String]): Unit = {

		//Create spark session with the name 'Clustering in Archive Collection' that runs locally on all the core 
		    val spark = SparkSession.builder()
				//.master("local[*]")
				.appName("Clustering in Archeive Collection")
				//.config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse")
				.config("spark.sql.warehouse.dir", "temp/spark-warehouse")
				.getOrCreate()

				//RDD[Row(timeStemp, fileName, fileContent)]
				//File name is only the time stamp stamp
				//val trainingData = spark.sparkContext.wholeTextFiles("data/KDDTraining")
				val trainingData = spark.sparkContext.wholeTextFiles("data/news/uk/UK_1K")

				//Pre-process Dataset
				val preProcessedTrainingData = preProcessData(spark, trainingData)
				
				preProcessedTrainingData.cache()

				//Computes LDA models hierarchy
				computeLDAModelsAndTopicsHierarchy(spark, preProcessedTrainingData)

				//Prints topics hierarchy
				printTopicsHierarchy(spark)
				
//				//print combined view of first and second level topics
//				println("Preparing to print first and second level topics dataframe.....")
//				computeCombinedViewOfFirstAndSecondLevelTopics(spark)
				
				spark.stop()

	}

	//Pre-processing data: RDD[(String, String)]
	def preProcessData(spark: SparkSession, data: RDD[(String, String)]): Dataset[Row] = {

	  /*
	   * RDD[Row(timeStemp, fileName, fileContent)]
	   * Filename has been composed of TimeStamp and Filename. Separator as a "-" has been used
	   * Data pre-processing steps includes:
	   * 	1. Remove all the words that has length 1 or 2
	   * 	2. Perform Lematization
	   * 	2. Keep only alphabetic characters and space
	   * 	3. Replace all the spaces characters [\t\n\r\f]
	   * 	4. Convert all characters to lower case
	   * 
	   */
				val tempData = data.map(kvTouple => Row(
				//DateTimeFormat.forPattern("YYYY-MM-dd").print(kvTouple._1.split(".")(0).toLong),
				kvTouple._1,
				"FileName",
				plainTextToLemmas(kvTouple._2.replaceAll("""[^a-zA-Z0-9\s]""", ""), ENGLISH_STOP_WORDS_LIST)
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

//		//Remove stop words
//		val stopWordsRemoved = new StopWordsRemover()
//		.setInputCol("words")
//		.setOutputCol("stopWordsFiltered")
//		//.setStopWords(StopWordsRemover.loadDefaultStopWords("english"))
//		.transform(wordsData)

		return wordsData
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
		//tfModelF.write.overwrite().save("LDAModels/featuresSpace/firstLevelTF")

		val featurizedDataF  = tfModelF.transform(preProcessedDataset)

		//Train LDA Model
		val ldaModelF = new LDA()
		.setK(NUMBER_OF_FIRST_LEVEL_TOPICS)
		.setMaxIter(FIRST_LEVEL_MAX_ITR)
		.setFeaturesCol("featuresTFF")
		.setTopicDistributionCol("topicDistributionF")
		.fit(featurizedDataF)

		//save LDA model
		//ldaModelF.write.overwrite().save("LDAModels/firstLeveTopiclModel")

		//Save as temp view
		//ldaModelF.transform(featurizedDataF).createOrReplaceTempView("firstLevelTopicsView")

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
		//topicsDFF.createOrReplaceTempView("firstLevelTopicsView")
		
		FIRST_LEVEL_TOPIC_DF = topicsDFF

		/*--------------Work towards second level models and topics----------------*/

		//Filter out each topic and documents in it and store them in topicsDFs array
		val transDataSet = ldaModelF.transform(featurizedDataF)
		
		transDataSet.cache()
		
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
			//tfModelS.write.overwrite().save("LDAModels/featuresSpace/secondLevelTF"+i)

			val featurizedDataS  = tfModelS.transform(TOPIC_DOCUMENTS_DFs(i))

			//Train LDA Model
			val ldaModelS = new LDA()
			.setK(NUMBER_OF_SECOND_LEVEL_TOPICS)
			.setMaxIter(SECOND_LEVEL_MAX_ITR)
			.setFeaturesCol("featuresTFS")
			.setTopicDistributionCol("topicDistributionS")
			.fit(featurizedDataS)

			//save LDA model
			//ldaModelS.write.overwrite().save("LDAModels/secondLeveTopicsModel"+i)

			//save the transformation as a temp view
			//ldaModelS.transform(featurizedDataS).createOrReplaceTempView("secondLevelTopicsView"+i)

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
			//topicsDFS.createOrReplaceTempView("secondLevelTopicsView"+i)
			
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
	    
	}
	
	//computes combined view of first and second level topics
	def computeCombinedViewOfFirstAndSecondLevelTopics(spark: SparkSession): Unit = {
	  
	  //Define the schema of dataframe and should look like:  index, firstLevelTopic, secondLevelTopic
	  
	  val schemaString = "index firstLevelTopic secondLevelTopic"
	  
	  val dfSchema = StructType(schemaString.split(" ").map { fieldName => StructField(fieldName, StringType, true) })
	  
	  //val emptyDS = spark.createDataFrame(spark.sparkContext.emptyRDD[Row], dfSchema)
	  
	  //var resultantDS = emptyDS
	  
	  var tempList: List[Row] = new ArrayList[Row]() //will be used to push row into DataFrame
	  
	  println("Line 303")
	  
	  FIRST_LEVEL_TOPIC_DF.foreach { fRow => {
	    
	    var flTopicString = ""
	    fRow.getAs[WrappedArray[String]]("firstLevelTopic").toArray.foreach { itemF => flTopicString + itemF +" " }
	    val flIndex = fRow.getAs[Integer]("topic")
	    
	    println(s"Line 311:  $flIndex")
	    SECONG_LEVEL_TOPICS_DFs(flIndex).foreach { sRow => {
	      
	      println("Line 314")
	      var slTopicString = ""
	      sRow.getAs[WrappedArray[String]]("secondLevelTopic").toArray.foreach{ itemS => slTopicString = slTopicString + itemS + " "}
	      
	      println("Line 318")
	      
	      val topicRow = Row(flIndex, flTopicString, slTopicString)
	      tempList.add(topicRow)
	      
//	      //convert topicRow to DataFrame
//	      val tempDF = spark.sqlContext.createDataFrame(tempList, dfSchema)
//	      
//	      println("Line 326")
//	      
//	      //Insert into resultant DataFrame
//	      resultantDS = resultantDS.union(tempDF)
	      
//	      tempList.clear()
	      
	    } }
	     
	  } }
	  
	  //convert tempList to DataFrame
	  val finalDF = spark.sqlContext.createDataFrame(tempList, dfSchema)
	  
	  //Save the final DataFrame to disk
	  finalDF.rdd.repartition(1).saveAsTextFile("data/LDATopics")
	  
	  finalDF.show()
	  
	  tempList.clear
	  
	}
	/*
	 *Lemmatization, stop words and wordsLength < 3 removal
	 */
	def plainTextToLemmas(text: String, stopWords: Set[String]): String = {
		val props = new Properties()
		props.put("annotators", "tokenize, ssplit, pos, lemma")
		val pipeline = new StanfordCoreNLP(props)
		val doc = new Annotation(text)
		pipeline.annotate(doc)
		val lemmas = new ArrayBuffer[String]()
		val sentences = doc.get(classOf[SentencesAnnotation])
		for (sentence <- sentences; token <- sentence.get(classOf[TokensAnnotation])) {
			val lemma = token.get(classOf[LemmaAnnotation])
					if (lemma.length > 2 && !stopWords.contains(lemma)) {
						lemmas += lemma.toLowerCase
					}
		}
		
		var finalString: String = ""
		
		lemmas.foreach { item => finalString = finalString + item + " " }
		
		finalString.trim()
	}
}

/*
 *Very important piece of code
 * 
 * To extract individual field from document 
 * Document structure:
 * 	timestamp: 123456
 * 	title: some title goes here
 * 	text: this is the main content of document. This also includes new lines
 * 
 * val temp = "timestamp: 1234567\ntitle: some title goes here\ntext: "main content of document goes here\nsome text goes here\nsome more text goes here"
        
    val result = temp.split("\n")
    
    val timeStamp = result(0).split(": ")(1)
    val title = result(1).split(": ")(1)
    
    val arrayLen = result.length
    var text = result(2).split(": ")(1)
    
    if (arrayLen > 2){
      for(index <- 3 until arrayLen){
        text = text+ " " + result(index)
      }
    }
    
    println(s"TIMESTAMP: $timeStamp")
    println(s"TITLE: $title")
    println(s"TEXT: $text") 
 */















