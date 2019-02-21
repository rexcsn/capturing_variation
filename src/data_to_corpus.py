import argparse
import json
import multiprocessing
import sys
from collections import defaultdict
from itertools import islice, count
import pandas as pd
import numpy as np
import bz2
import string
import re

# project_path = "/shared/0/projects/location-inference/working-dir/textual_data"
project_path = "."


def extract_info(filename):
    labels = ['user', 'text', 'id', 'source', 'place', 'geo']
    file = bz2.BZ2File(filename)
    collect = []
    for line in file:
        all_present = True
        js = json.loads(line)
        for item in labels:
            if js[item] == "null":
                all_present = False
                break
        if all_present:
            collect.append(js)

    df = pd.DataFrame(collect)
    df = df[labels]

    filtered = pd.DataFrame()

    filtered = filtered.assign(user_id=df['user'].apply(pd.Series)['id'],
                               tweet_id=df['id'],
                               text=df['text'],
                               lat=df['geo'].apply(pd.Series)['coordinates'][0],
                               lon=df['geo'].apply(pd.Series)['coordinates'][1],
                               city=df['place'].apply(pd.Series)['full_name'],
                               country_code=df['place'].apply(pd.Series)['country_code'],
                               source=df['source']
                               )

    filtered.to_csv(
        "%s/extracted_2019-02-02" % project_path)
    print("Saved df to %s/extracted_2019-02-02 !!!" % project_path)

    return filtered


def text_to_words(df):
    for item in string.punctuation:
        df.str.replace(item, ' ')

    return df.str.split()


def to_corpus(df):
    dict_df = pd.DataFrame()
    dict_df = dict_df.assign(tags=(df['city']+","+df['country_code']).str.replace(' ', ''),
                             words=df['text'].str.split(r'\W+'))

    dict_df.to_json(
        "%s/sample_corpus_2019-02-02" % project_path,
        orient='records')
    print("Saved corpus to %s/sample_corpus_2019-02-02 !!!" % project_path)

    return dict_df


def main():
    # df = extract_info("/twitter-turbo/decahose/raw/decahose.2019-02-02.p1.bz2")
    # df = extract_info("./sample_text.txt.bz2")
    df = extract_info("./baby.txt.bz2")
    to_corpus(df)


if __name__ == "__main__":
    main()
