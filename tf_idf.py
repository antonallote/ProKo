
from pyspark import RDD


def calc_word_doc_pairs(data:RDD[tuple[str, list[str]]])->RDD[tuple[tuple[str,str],int]]:
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

    word_doc_unique = data.flatMap(
        lambda x: [(word, x[0]) for word in set(x[1])]
    )
    print('Counts how many docs each word appears in.\nExample: [("hi", "file1"), ("hi", "file2")] → [("hi", 2)]')
    df = (word_doc_unique
          .map(lambda x: (x[0], 1))
          .reduceByKey(lambda a, b: a + b))
    return df

def calc_tf_idf(tf:RDD[tuple[tuple[str,str],int]],idf:dict[str,float]):
    return tf.map(lambda x: ((x[0][0], x[0][1]), x[1] * idf[x[0][0]]))

