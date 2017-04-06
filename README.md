# Abstract
Clustering real time text data streams is an important issue in industries as well as in data mining community. Many applications require real time clustering of text streams such as text crawling, document organization, news filtering and topic detection & tracking etc. In this study, we have implemented a method for real time clustering of text streams. We maintain a two level hierarchy of (sub)topics over the news stream. First level models the main topics of global importance. Each main topic can be more precisely captured through a set of subtopics, which we called second level topics. Specifically, and informally, we deal with the following problem in this study:

"How can we learn, maintain and detect bursty and emerging (sub)topics over stream of news articles ?"

# Tools and Technologies (Scala project):
  1. Spark with Scala 
  2. Spark Streaming
  3. Spark ML: K-Means, LDA, PCA, word2vec, TFIDF
  4. Spark SQL, Dataframe, UDF
  5. HDFS
  6. Yarn
  7. Eclipse
