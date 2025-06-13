from pathlib import Path
from typing import Optional

from pyspark import SparkConf, SparkContext
import math
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import LDA

from pyspark.sql import SparkSession
from pyspark import RDD


#setup
conf = SparkConf().setMaster("local[*]")
sc = SparkContext.getOrCreate(conf=conf)
spark = SparkSession.builder \
    .config(conf=sc.getConf()) \
    .getOrCreate()
collect_while_dev= True

def load_data(path:str)-> RDD:
    """
    :param path: where the data is located
    :type path: str
    :return: RDD of documents
    """

    return sc.wholeTextFiles(path)

def preprocess(data:RDD[tuple[str, list[str]]], collect:Optional[bool] = False, subset:Optional[int]=None)->RDD[tuple[str, list[str]]]| list[tuple[str,list[str]]]:
    """
    preprocess the data by splitting words via space and make em all lowercase. Has few other Options that are helpful for dev

    :param data: the dataset to be preprocessed
    :param collect: whether to collect the whole dataset or not, default is False
    :param subset: if you dont want to collect the whole dataset, you can give how many documents should be collected as subset
    :return: if not specified returns the lazy RDD else a list of tuples where the first entry is the filepath as string and the second entry is a list of strings ( all words of the corresponding document in lowercase)
    """
    if collect and subset is not None:
        raise ValueError("Either you want to collect the whole dataset or subset should be specified both cant be set")

    splitted_data = data.mapValues(lambda text: text.lower().split())

    if collect:
        return splitted_data.collect()
    if subset:
        return splitted_data.take(subset)
    else:
        return  splitted_data

def word_doc_pairs(data:RDD[tuple[str, list[str]]])->RDD[tuple[tuple[str,str],int]]:
    """
    :param data: data where each entry looks like this: ('document_path', ['w1', 'w2', …])
    :return: flattend data where each entry looks like this:  ((w1, document_path), 1)
    """
    return data.flatMap(
        lambda x: [((word, x[0]), 1) for word in x[1]]
    )
def calc_tf(word_doc_pairs:RDD[tuple[tuple[str,str],int]])->RDD[tuple[tuple[str,str],int]]:
    #(('peer', 'file:/Users/antonvolker/Master_AI/ProKo/texts/Dutch/Baas Gansendonck - Hendrik Conscience.txt'), 8)
    #Sums for each key (word,doc) the values ( its always 1 so this is a count operation) -> term frequency per doc it is if you wanna be super precise
    return word_doc_pairs.reduceByKey(lambda a, b: a + b)

def calc_df(data:RDD[tuple[str, list[str]]])->RDD[tuple[str,int]]:

    print('Creates (word, doc) pairs from each (doc, words), removing duplicates.\nExample: ("file.txt", ["hi", "world", "hi"]) → [("hi", "file.txt"), ("world", "file.txt")]')

    word_doc_unique = docs_subset.flatMap(
        lambda x: [(word, x[0]) for word in set(x[1])]
    )
    print('Counts how many docs each word appears in.\nExample: [("hi", "file1"), ("hi", "file2")] → [("hi", 2)]')
    df = (word_doc_unique
          .map(lambda x: (x[0], 1))
          .reduceByKey(lambda a, b: a + b))
    return df

if __name__ == '__main__':

    rdd = load_data(path="./texts/**/*.txt")
    docs_list = preprocess(rdd,subset=1)
    docs_subset = sc.parallelize(docs_list) # TODO: findout why this is done / move into preprocess

    # Document Frequency
    df = calc_df(data=docs_subset)

    # Term Frequency
    word_doc_pairs = word_doc_pairs(docs_subset)
    tf = calc_tf(word_doc_pairs=word_doc_pairs)


    # Inverse Document Frquency log(Number of Documents / Document Frequency)
    n_docs = docs_subset.count()
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