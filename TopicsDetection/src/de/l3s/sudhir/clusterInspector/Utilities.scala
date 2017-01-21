package de.l3s.sudhir.clusterInspector

/**
 * This object provides with helper API
 */
object Utilities_Inspector {
  
  
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