from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local[*]")

sc = SparkContext.getOrCreate(conf=conf)

if __name__ == '__main__':
    rdd = sc.wholeTextFiles("./texts/**/*.txt")
    docs = rdd.mapValues(lambda text: text.lower().split())

    print(docs.take(1))

    pass