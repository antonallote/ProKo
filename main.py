import math
import time

from lda import build_vocabulary, build_document_dataframe, train_lda, describe_topics
from tf_idf import calc_df, calc_tf, calc_word_doc_pairs, calc_tf_idf
from util import load_data, preprocess

from pyspark import SparkConf, SparkContext

from pyspark.sql import SparkSession
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

stopword_map = {
    'English': set(stopwords.words('english')),
    'German': set(stopwords.words('german')),
    'French': set(stopwords.words('french')),
    'Spanish': set(stopwords.words('spanish')),
    'Italian': set(stopwords.words('italian')),
    'Dutch': set(stopwords.words('dutch')),
}


#setup
conf = SparkConf().setMaster("local[*]")
sc = SparkContext.getOrCreate(conf=conf)
spark = SparkSession.builder \
    .config(conf=sc.getConf()) \
    .getOrCreate()

def pipeline():
    start = time.time()

    rdd = load_data(path="./texts/**/*.txt", sc=sc)
    docs_list = preprocess(rdd, stopword_map=stopword_map)
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

    # Print
    # for ((word, doc), score) in tf_idf.collect():
    #    print(f"{word} in {doc} → TF-IDF: {score:.4f}")

    vocab, word_to_index = build_vocabulary(idf_dict)
    vocab_size = len(vocab)

    documents_df = build_document_dataframe(tf_idf, word_to_index, vocab_size, spark)

    model = train_lda(documents_df, num_topics=10, max_iter=10)
    topics_df = describe_topics(model, num_words=10)

    topics_df.show(truncate=False)

    # postprocessing
    index_to_word = {index: word for word, index in word_to_index.items()}
    from pyspark.sql.functions import udf
    from pyspark.sql.types import ArrayType, StringType

    def decode_indices(indices: list[int]) -> list[str]:
        return [index_to_word.get(i, f"<unk_{i}>") for i in indices]

    decode_udf = udf(decode_indices, ArrayType(StringType()))
    decoded_topics = topics_df.withColumn("termWords", decode_udf(topics_df["termIndices"]))
    decoded_topics.select("topic", "termWords", "termWeights").show(truncate=False)
    duration = time.time() - start


    from pyspark.sql.functions import concat_ws

    # Arrays zu Strings konvertieren
    decoded_topics_csv_ready = decoded_topics \
        .withColumn("termWordsStr", concat_ws(",", "termWords")) \
        .withColumn("termWeightsStr", concat_ws(",", "termWeights")) \
        .select("topic", "termWordsStr", "termWeightsStr")

    (decoded_topics_csv_ready
        .coalesce(1) \
        .write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv("./output/topics"))

    with open("./output/duration.txt", "w") as f:
        f.write(f"Pipeline duration: {duration:.2f} seconds\n")



if __name__ == '__main__':
    pipeline()
    pass

