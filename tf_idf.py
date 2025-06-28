from pyspark import RDD


def calc_word_doc_pairs(data: RDD[tuple[str, list[str]]]) -> RDD[tuple[tuple[str, str], int]]:
    """
    Generate word-document pairs from the input RDD.

    Each document's list of words is flattened so that for each word occurrence in a document,
    a pair ((word, document_path), 1) is created.

    :param data: RDD with entries like ('document_path', ['word1', 'word2', ...])
    :return: RDD of ((word, document_path), 1) for each word occurrence
    """
    return data.flatMap(
        lambda x: [((word, x[0]), 1) for word in x[1]]
    )


def calc_tf(word_doc_pairs: RDD[tuple[tuple[str, str], int]]) -> RDD[tuple[tuple[str, str], int]]:
    """
    Compute the term frequency (TF) for each word in each document.

    It sums the occurrences of each word per document.

    :param word_doc_pairs: RDD with entries like ((word, document_path), 1)
    :return: RDD with entries like ((word, document_path), frequency)
    """
    return word_doc_pairs.reduceByKey(lambda a, b: a + b)


def calc_df(data: RDD[tuple[str, list[str]]]) -> RDD[tuple[str, int]]:
    """
    Compute document frequency (DF) for each word — i.e., in how many documents each word appears.

    It deduplicates word-document pairs before counting.

    :param data: RDD with entries like ('document_path', ['word1', 'word2', ...])
    :return: RDD with entries like (word, number_of_documents_containing_word)
    """
    print('Creates (word, doc) pairs from each (doc, words), removing duplicates.\n'
          'Example: ("file.txt", ["hi", "world", "hi"]) → [("hi", "file.txt"), ("world", "file.txt")]')

    word_doc_unique = data.flatMap(
        lambda x: [(word, x[0]) for word in set(x[1])]
    )

    print('Counts how many docs each word appears in.\n'
          'Example: [("hi", "file1"), ("hi", "file2")] → [("hi", 2)]')

    df = (word_doc_unique
          .map(lambda x: (x[0], 1))
          .reduceByKey(lambda a, b: a + b))
    return df


def calc_tf_idf(tf: RDD[tuple[tuple[str, str], int]], idf_broadcast) -> RDD[tuple[tuple[str, str], float]]:
    """
    Compute TF-IDF score for each word-document pair.

    Multiplies term frequency (TF) with the corresponding inverse document frequency (IDF).

    :param tf: RDD with entries like ((word, document_path), term_frequency)
    :param idf: Dictionary with entries like {word: idf_score}
    :return: RDD with entries like ((word, document_path), tf_idf_score)
    """
    return tf.map(lambda x: (x[0], x[1] * idf_broadcast.value.get(x[0][0], 0.0)))



