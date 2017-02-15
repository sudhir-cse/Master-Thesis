package de.l3s.sudhir.stemming

import org.apache.spark.sql.functions._

object Test {
  
  def main(args:Array[String]):Unit = {
    
    
    
  }
  
 /**
* User defined functions
* 
* computes total number of words with each document to a new column
*/
val totalWords = udf[Long, Seq[String]]( words => {

    words.length
  
  } 
)
  
}