from pathlib import Path
from typing import Optional

from pyspark import SparkConf, SparkContext
import math
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import LDA

from pyspark.sql import SparkSession


if __name__ == '__main__':



    pass
    ### LDA



    # # 1. Build a vocabulary and assign each word an index
    # #    (assumes you have an RDD of (word,tf-idf_dict) or you can extract from idf_dict)
    # vocab = sorted(idf_dict.keys())
    # word2idx = {w: i for i, w in enumerate(vocab)}
    # vocabSize = len(vocab)
    #
    # # 2. From your tf_idf RDD: [ ((word, doc), score) ] → RDD[(doc, (word,score))]
    #
    # doc_word_scores = tf_idf.map(
    #     lambda x: (
    #         # x[0] == (word, doc) → x[0][1] is doc
    #         x[0][1],
    #         # x[0][0] is word, x[1] is score
    #         (x[0][0], x[1])
    #     )
    # )
    #
    #
    # def to_row(doc_ws):
    #     doc_id, word_score_iter = doc_ws
    #     # 1. Materialize the iterable into a list
    #     ws_list = list(word_score_iter)  # [(word, score), ...]
    #     # 2. Turn into (index, score) pairs and sort by index
    #     idx_score = sorted(
    #         ((word2idx[word], score) for word, score in ws_list),
    #         key=lambda pair: pair[0]
    #     )
    #     if idx_score:
    #         indices, values = zip(*idx_score)
    #     else:
    #         indices, values = [], []
    #     # 3. Build the SparseVector with sorted indices
    #     vec = Vectors.sparse(vocabSize, list(indices), list(values))
    #     return Row(docId=doc_id, features=vec)
    #
    #
    # doc_vecs = doc_word_scores \
    #     .groupByKey() \
    #     .map(to_row)
    #
    # df = spark.createDataFrame(doc_vecs)
    # # 5. Fit LDA
    # lda = LDA(k=10, maxIter=20, featuresCol="features", seed=42)
    # model = lda.fit(df)
    #
    # # 6. Inspect topics
    # topics = model.describeTopics()
    # topics.show(truncate=False)
    #
    # pass