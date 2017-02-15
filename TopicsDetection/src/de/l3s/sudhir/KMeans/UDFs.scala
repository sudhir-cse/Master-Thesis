package de.l3s.sudhir.KMeans

import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.SparseVector

object UDFs {
  
  /*---------------User defined functions------------*/

//remove all the garbage words
//val stopWords = Seq("going","its","this","make","error","pay","biggest","hot", "just", "average","imf", "que", "los", "del", "las", "welsh", "large", "quite", "little", "great", "programme", "second", "past")

val ENGLISH_STOP_WORDS_LIST  = Seq("going","its","this","make","error","pay","biggest","hot", "just", "average","imf", "que", "los", "del", "las", "welsh", "large", "quite", "little", "great", "programme", "second", "past","a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the")


 val filterStopWords = udf[Seq[String], Seq[String]]( words => {
      words.filter(word => !ENGLISH_STOP_WORDS_LIST.contains(word) )
    } 
)

//computes total number of words with each document to a new column
val totalWords = udf[Long, Seq[String]]( words => {
      words.length  
    } 
)

//compute
val distinctWordCount = udf[Int, Seq[String]]( words =>
    words.distinct.length
)

//val computeSQD = udf[Double, Int, SparseVector]( (clusterIndex, tfidf) => {
//    
//      var distance: Double = 0.0
//      val tfidfArray = tfidf.toArray
//      val clusterCenter = firstLevelClustersCrenter(clusterIndex)
//      
//      //compute squire distance
//      if(tfidfArray.size == clusterCenter.size){
//        for(index <- 0 until tfidfArray.size){
//          distance = distance + (tfidfArray(index) - clusterCenter(index)) * (tfidfArray(index) - clusterCenter(index))
//        }
//      }   
//      distance
//    } 
//  )
  
}