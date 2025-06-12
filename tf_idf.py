from pyspark import SparkConf, SparkContext
import math
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import LDA

from pyspark.sql import SparkSession


conf = SparkConf().setMaster("local[*]")
sc = SparkContext.getOrCreate(conf=conf)
spark = SparkSession.builder \
    .config(conf=sc.getConf()) \
    .getOrCreate()
collect_while_dev= True
if __name__ == '__main__':
    rdd = sc.wholeTextFiles("./texts/**/*.txt")
    docs_list = rdd.mapValues(lambda text: text.lower().split()).take(2)
    print("docs_subset is List[Tuple[str,List[str]] , where each elem of the list is a tuple consisting of the filepath and the words from this file in a list of strings")
    docs_subset = sc.parallelize(docs_list)

    # Term Frequency
    print("word_pairs is List[Triplet[word,document,1]]")
    word_doc_pairs = docs_subset.flatMap(
        lambda x: [((word, x[0]), 1) for word in x[1]]
    )

    print("tf is List[Triplet[word,document,count]]")
    tf = word_doc_pairs.reduceByKey(lambda a, b: a + b)


    # Document Frequency
    print('Creates (word, doc) pairs from each (doc, words), removing duplicates.\nExample: ("file.txt", ["hi", "world", "hi"]) → [("hi", "file.txt"), ("world", "file.txt")]')

    word_doc_unique = docs_subset.flatMap(
        lambda x: [(word, x[0]) for word in set(x[1])]
    )

    print('Counts how many docs each word appears in.\nExample: [("hi", "file1"), ("hi", "file2")] → [("hi", 2)]')
    df = (word_doc_unique
          .map(lambda x: (x[0], 1))
          .reduceByKey(lambda a, b: a + b))


    print("tf_idf")
    # 1. Total number of documents
    n_docs = docs_subset.count()

    # 2. Compute IDF for each word: idf = log(n_docs / df)
    idf = df.mapValues(lambda df_val: math.log(n_docs / df_val))

    # 3. Convert IDF to a dictionary so we can join manually (broadcast-like)
    idf_dict = dict(idf.collect())

    # 4. Compute TF-IDF: tf_idf = tf * idf
    tf_idf = tf.map(lambda x: ((x[0][0], x[0][1]), x[1] * idf_dict[x[0][0]]))
    # x[0][0] = word, x[0][1] = doc, x[1] = tf count

    # 5. Output result
    for ((word, doc), score) in tf_idf.collect():
        print(f"{word} in {doc} → TF-IDF: {score:.4f}")



    ### LDA



    # 1. Build a vocabulary and assign each word an index
    #    (assumes you have an RDD of (word,tf-idf_dict) or you can extract from idf_dict)
    vocab = sorted(idf_dict.keys())
    word2idx = {w: i for i, w in enumerate(vocab)}
    vocabSize = len(vocab)

    # 2. From your tf_idf RDD: [ ((word, doc), score) ] → RDD[(doc, (word,score))]

    doc_word_scores = tf_idf.map(
        lambda x: (
            # x[0] == (word, doc) → x[0][1] is doc
            x[0][1],
            # x[0][0] is word, x[1] is score
            (x[0][0], x[1])
        )
    )


    def to_row(doc_ws):
        doc_id, word_score_iter = doc_ws
        # 1. Materialize the iterable into a list
        ws_list = list(word_score_iter)  # [(word, score), ...]
        # 2. Turn into (index, score) pairs and sort by index
        idx_score = sorted(
            ((word2idx[word], score) for word, score in ws_list),
            key=lambda pair: pair[0]
        )
        if idx_score:
            indices, values = zip(*idx_score)
        else:
            indices, values = [], []
        # 3. Build the SparseVector with sorted indices
        vec = Vectors.sparse(vocabSize, list(indices), list(values))
        return Row(docId=doc_id, features=vec)


    doc_vecs = doc_word_scores \
        .groupByKey() \
        .map(to_row)

    df = spark.createDataFrame(doc_vecs)
    # 5. Fit LDA
    lda = LDA(k=10, maxIter=20, featuresCol="features", seed=42)
    model = lda.fit(df)

    # 6. Inspect topics
    topics = model.describeTopics()
    topics.show(truncate=False)

    pass