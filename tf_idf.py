from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local[*]")

sc = SparkContext.getOrCreate(conf=conf)
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
          .distinct()
          .map(lambda x: (x[0], 1))
          .reduceByKey(lambda a, b: a + b))

    pass