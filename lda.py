from pyspark import RDD
from pyspark.sql import Row, SparkSession, DataFrame
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import LDA
from typing import Tuple, List, Dict


def build_vocabulary(idf: Dict[str, float]) -> Tuple[List[str], Dict[str, int]]:
    """
    Build a sorted vocabulary and a mapping from word to index.

    :param idf: Dictionary mapping words to IDF scores.
    :return: A tuple containing the sorted vocabulary list and a word-to-index mapping.
    """
    vocabulary = sorted(idf.keys())
    word_to_index = {word: idx for idx, word in enumerate(vocabulary)}
    return vocabulary, word_to_index


def reshape_tf_idf(tf_idf: RDD[Tuple[Tuple[str, str], float]]) -> RDD[Tuple[str, Tuple[str, float]]]:
    """
    Reshape TF-IDF data for grouping by document.

    :param tf_idf: RDD of ((word, document), tf_idf_score)
    :return: RDD of (document, (word, score))
    """
    return tf_idf.map(lambda x: (x[0][1], (x[0][0], x[1])))


def to_sparse_vector(doc: Tuple[str, List[Tuple[str, float]]],
                     word_to_index: Dict[str, int],
                     vocab_size: int) -> Row:
    """
    Convert a document's word-score pairs to a SparseVector Row.

    :param doc: Tuple of (document_id, [(word, score), ...])
    :param word_to_index: Mapping from word to vocabulary index.
    :param vocab_size: Total number of words in the vocabulary.
    :return: A Row with docId and SparseVector features.
    """
    doc_id, word_scores = doc
    indexed = sorted(
        ((word_to_index[word], score) for word, score in word_scores if word in word_to_index),
        key=lambda x: x[0]
    )
    if indexed:
        indices, values = zip(*indexed)
    else:
        indices, values = [], []
    vector = Vectors.sparse(vocab_size, list(indices), list(values))
    return Row(docId=doc_id, features=vector)


def build_document_dataframe(tf_idf: RDD[Tuple[Tuple[str, str], float]],
                             word_to_index: Dict[str, int],
                             vocab_size: int,
                             spark: SparkSession) -> DataFrame:
    """
    Create a DataFrame with documents represented as sparse TF-IDF vectors.

    :param tf_idf: RDD of ((word, document), tf_idf_score)
    :param word_to_index: Mapping from word to index in the vocabulary.
    :param vocab_size: Number of unique words in the vocabulary.
    :param spark: The active SparkSession.
    :return: DataFrame with columns: docId (str), features (SparseVector)
    """
    grouped = reshape_tf_idf(tf_idf).groupByKey()
    rows = grouped.map(lambda doc: to_sparse_vector(doc, word_to_index, vocab_size))
    return spark.createDataFrame(rows)


def train_lda(documents: DataFrame,
              num_topics: int = 10,
              max_iter: int = 20,
              seed: int = 42) -> LDA:
    """
    Train an LDA model on the given document feature DataFrame.

    :param documents: DataFrame with columns: docId, features (SparseVector).
    :param num_topics: The number of topics to infer.
    :param max_iter: Maximum number of EM iterations.
    :param seed: Random seed for reproducibility.
    :return: Trained LDA model.
    """
    lda = LDA(k=num_topics, maxIter=max_iter, featuresCol="features", seed=seed)
    return lda.fit(documents)


def describe_topics(model: LDA, num_words: int = 10) -> DataFrame:
    """
    Return the top words for each topic in the LDA model.

    :param model: Trained LDA model.
    :param num_words: Number of top words to show per topic.
    :return: DataFrame with topics and top word indices.
    """
    return model.describeTopics(num_words)
