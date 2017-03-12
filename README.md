# Master-Thesis
Text stream clustering algorithm that detects, tracks and updates large and small burst of news in two-level topic hierarchy

#Tools and Technologies (Scala project):
  1. Spark / Scala 
  2. Spark Streaming
  3. Spark ML: K-Means, LDA, PCA, word2vec, TFIDF
  4. Spark SQL
  5. HDFS
  6. Yarn
  7. Eclipse

#Abstract
Clustering real time text data streams is an important issue in data mining community. Many applications require real time clustering of text streams such as text crawling, document organization, news filtering and topic detection & tracking etc. In this study, we have implemented a method for real time clustering of text streams. We maintain a two level hierarchy of (sub)topics over the news stream. First level models the main topics of global importance. Each main topic can be more precisely captured through a set of subtopics, which we called second level topics. 

