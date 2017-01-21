/**
 * This object parse input files in a Directory and produce a single output file
 * 
 * Files format in input Directory:
 * 	Timestamp: 20111202203142
 * 	Title: BBC News - Mr Osborne's unwelcome statement
 * 	Text: Content of the file goes here
 * 	
 * Output file format:
 * 	each line:
 * 		<timestamp>20111202203142</timestamp><title>Title goes here</title><text>Content of file goes here</text>
 * 
 */

package de.l3s.sudhir.preProcessing

import java.text.SimpleDateFormat
import java.util.Date
import scala.collection.JavaConverters._
import scala.io.StdIn.{readLine,readInt}

import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import java.text.ParseException

object DataPreprocessor {
  
  def main(args: Array[String]): Unit = {
    
    //Read input and output files from terminal
    val INPUT_FILE = readLine("Please Provide the Input Directory: ")
    val OUTPUT_FILE = readLine("Please Provide the Output File: ")
    print("\nPlease Enter Minimum Number of Partition: ")
    val MIN_PARTITION = readInt()
    
    
    val spark = SparkSession.builder()
    .appName("Data Preprocessor")
    .config("spark.sql.warehouse.dir", "temp/spark-warehouse")
    .getOrCreate()
    
    val sc = spark.sparkContext
    
    ///user/sudhir/data/news/uk/UK_1K
    val inputData = sc.wholeTextFiles(INPUT_FILE, MIN_PARTITION)
    
    //RDD(Row(TimeStamp: string, title: String, title+content: String))
    val tdr1 = inputData.map(touple => {
      
      Row (
            Utilities1.extractFileFields(touple._2, "TIMESTAMP"),
            Utilities1.extractFileFields(touple._2, "TITLE"),
            Utilities1.filterOutStopWords((Utilities1.extractFileFields(touple._2, "TITLE") +" "+Utilities1.extractFileFields(touple._2, "FILECONTENT")).replaceAll("""[^a-zA-Z0-9\s]""", "").replaceAll("""\b\p{IsLetter}{1,2}\b""","").replaceAll("""\s+""", " "))
      )
      
    })
    
    //Remove records that 'error' in the date or content field
    val tdr2 = tdr1.filter { row => row.getAs[String](0) != "error" && row.getAs[String](2) != "error" }
    
    //Transform date string to java.utill.date
    val tdr3 = tdr2.map { row => Row (
      
        Utilities1.stringToDate(row.getAs[String](0)),
        row.getAs[String](1),
        row.getAs[String](2)
        
    ) }
    
    //Sort above RDD based on Date (first field) 
    val td3 = tdr3.sortBy(row => row.getAs[Date](0), true)
     
    /*
     * Transform: Row(textString)
     * textString: <timestamp>here goes timestamp</timestamp><title>here goes title</title><content>here goes title and file content</content>
     */
    val td4 = td3.map { row => "<timestamp>"+row.get(0)+"</timestamp>"+"<title>"+row.get(1)+"</title>"+"<content>"+row.get(2)+"</content>" }
    
    println("preprocessed data are being written to file");
    
    //save as text file
    td4.repartition(1).saveAsTextFile(OUTPUT_FILE)
    
    spark.stop()
    
  }
  
}//end of DataPreprocessor


object Utilities1 {
  
  /*
	 * This method is used to extract file fields (time-stamp, title and content)
	 * 
	 */
  def extractFileFields(fileText: String, fieldName: String): String = {
    
    var result = "error"
    
    try {
      val splitFileText = fileText.split("\n")
      
      if(fieldName.equalsIgnoreCase("TIMESTAMP")){
        
        if(splitFileText(0).split(": ").length == 2 && splitFileText(0).split(": ")(1)!=null){
        
          val dateString = splitFileText(0).split(": ")(1)
         
          dateString.toLong
          //Date format- 1996-12-25 02:58:09
          result = dateString(0).toString+dateString(1).toString+dateString(2).toString()+dateString(3).toString()+"-"+dateString(4).toString()+dateString(5).toString()+"-"+dateString(6).toString()+dateString(7).toString()+" "+dateString(8).toString()+dateString(9).toString()+":"+dateString(10).toString()+dateString(11).toString()+":"+dateString(12).toString()+dateString(13).toString()
          stringToDate(result)
         
         }
        }
        
      else if(fieldName.equalsIgnoreCase("TITLE")){
        
        if(splitFileText(1).split(": ").length == 2 && splitFileText(1).split(": ")(1) != null)
          result = splitFileText(1).split(": ")(1)
      }
        
      else if(fieldName.equalsIgnoreCase("FILECONTENT")){
        val arrayLen = splitFileText.length
        
        if(splitFileText(2).split(": ").length == 2 && splitFileText(2).split(": ")(1) != null)
          result = splitFileText(2).split(": ")(1)
        
        if(arrayLen > 3){
          for(index <- 3 until arrayLen)
            result = result+ " " + splitFileText(index)
        }
      }
      else{}
    }
    
   catch{
        case nf: NumberFormatException => { result = "error" } //nad: not a date
        case siob: StringIndexOutOfBoundsException => {result = "error"}
        case pe: ParseException => { result = "error" }
        case e: Exception => {result = "error"}
      }
    
    result
  }
  
  /*
   * Parse date in string to Date object
   */
  def stringToDate(dateInString: String): Date = {
    
      val sdf = new SimpleDateFormat("yyyy-M-dd HH:mm:ss");
      val date = sdf.parse(dateInString);
      date
  }
   /*
    * Filter out stop-words
    * @Input: text string with a single white space as a separator between words
    * @Output: test string with stop word list removed
    */
  
  def filterOutStopWords(text: String): String = {
    
    val ENGLISH_STOP_WORDS_LIST  = Set("a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the").asJava;
    var result = "" 
    
    val notStopWords = text.split(" ").filter { word => ! ENGLISH_STOP_WORDS_LIST.contains(word) }
    
    notStopWords.foreach( word => result = result+word+" ")
    
    result
  }
  
}//end of Utilities1




