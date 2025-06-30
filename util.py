from typing import Optional

from pyspark import SparkConf, SparkContext
from pyspark import RDD

import pandas as pd
from pathlib import Path


def load_data(path:str,sc:SparkContext)-> RDD:
    '''
    :param sc: context providing method to load text files
    :type sc: SparkContext
    :param path: where the data is located
    :type path: str
    :return: RDD of documents
    '''

    return sc.wholeTextFiles(path)


from typing import Optional
from pyspark import RDD

def get_language(path: str) -> str:
    '''
    Extracts the language from the file path, assuming it's the parent folder name.
    Example: 'texts/German/myfile.txt' → 'German'
    '''
    return path.split('/')[-2]

def preprocess(data: RDD[tuple[str, str]],
               stopword_map: dict[str, set[str]],
               collect: Optional[bool] = False,
               subset: Optional[int] = None
              ) -> RDD[tuple[str, list[str]]] | list[tuple[str, list[str]]]:
    '''
    Preprocess the data: lowercase, split by whitespace, remove stopwords by language.

    :param data: RDD of (file_path, full_text)
    :param stopword_map: Dictionary mapping language → set of stopwords
    :param collect: Whether to collect the full dataset to the driver
    :param subset: Optional number of documents to take (only applies if not collecting)
    :return: Either a lazy RDD or a list of (file_path, list of preprocessed words)
    '''
    if collect and subset is not None:
        raise ValueError('Either collect the whole dataset or specify a subset — not both.')

    def clean_text(path: str, text: str) -> tuple[str, list[str]]:
        lang = get_language(path)
        stopwords = stopword_map.get(lang, set())
        words = text.lower().split()
        filtered = [w for w in words if w not in stopwords]
        return (path, filtered)

    cleaned = data.map(lambda x: clean_text(x[0], x[1]))

    if collect:
        return cleaned.collect()
    if subset:
        return cleaned.take(subset)
    return cleaned




def aggregate_res(path_to_results:str):





    csv_dir = path_to_results
    # Alle relevanten CSV-Dateien finden (ohne .crc oder andere Metadateien)
    csv_files = [f for f in csv_dir.glob("part-*.csv") if f.is_file()]

    # Alle CSV-Dateien einlesen und zu einem großen DataFrame zusammenfügen
    df_list = [pd.read_csv(f) for f in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    # Als eine gemeinsame CSV-Datei speichern
    combined_df.to_csv("./output/topics_combined.csv", index=False)

    print("Fertig: ./output/topics_combined.csv")

