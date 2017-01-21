# Master-Thesis
Text stream clustering algorithm that detects, tracks and updates large and small burst of news in two-level topic hierarchy

#Tools and Technologies (Scala project):
  1. Spark Streaming
  2. Spark ML: K-Means, LDA, PCA, word2vec, TFIDF
  2. Scala
  3. HDFS
  4. Yarn
  5. Eclipse

#Abstract
Clustering real time text data streams is an important issue in data mining community. Many applications require real time clustering of text streams such as text crawling, document organization, news filtering and topic detection & tracking etc. In this study, we have implemented a method for real time clustering of text streams. We maintain a two level hierarchy of topics over the stream collection the first level models the main topics of global importance for the collection. Each main topic can be more precisely captured through a set of subtopics, which we call second level topics, and are important locally, i.e., within the main topic. 

