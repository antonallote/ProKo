import math

from tf_idf import calc_df, calc_tf, calc_word_doc_pairs, calc_tf_idf
from util import load_data, preprocess

from pyspark import SparkConf, SparkContext

from pyspark.sql import SparkSession


#setup
conf = SparkConf().setMaster("local[*]")
sc = SparkContext.getOrCreate(conf=conf)
spark = SparkSession.builder \
    .config(conf=sc.getConf()) \
    .getOrCreate()
collect_while_dev= True



if __name__ == '__main__':

    rdd = load_data(path="./texts/**/*.txt")
    docs_list = preprocess(rdd,subset=5)
    docs_subset = sc.parallelize(docs_list)

    # Document Frequency
    df = calc_df(data=docs_subset)

    # Term Frequency
    word_doc_pairs = calc_word_doc_pairs(docs_subset)
    tf = calc_tf(word_doc_pairs=word_doc_pairs)

    # Inverse Document Frquency log(Number of Documents / Document Frequency)
    n_docs = docs_subset.count()
    idf = df.mapValues(lambda df_val: math.log(n_docs / df_val))

    # 3. Convert IDF to a dictionary so we can join manually (broadcast-like)
    idf_dict = dict(idf.collect())

    # 4. Compute TF-IDF: tf_idf = tf * idf
    tf_idf = calc_tf_idf(tf=tf, idf=idf_dict)

    # printout
    for ((word, doc), score) in tf_idf.collect():
        print(f"{word} in {doc} → TF-IDF: {score:.4f}")

