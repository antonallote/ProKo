from typing import Optional

from pyspark import SparkConf, SparkContext
from pyspark import RDD




def load_data(path:str,sc:SparkContext)-> RDD:
    """
    :param sc: context providing method to load text files
    :type sc: SparkContext
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