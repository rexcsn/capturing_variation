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

# project_path = "/shared/0/projects/location-inference/working-dir/textual_data"
project_path = "."


def extract_info(filename):
    file = bz2.BZ2File(filename)
    df = pd.DataFrame()
    collect = []
    for line in file:
        dict = json.loads(line)
        collect = collect.append(dict)

    df = pd.DataFrame(collect)
    df = df[['user', 'text', 'id', 'source', 'place', 'geo']]
    print(np.shape(df))
    exit(1)


    filtered = pd.DataFrame(
        columns=['user_id', 'tweet_id', 'text', 'lat', 'lon', 'city', 'country_code', 'source'])

    filtered.assign(user_id=df['user']['id'])
    filtered.assign(tweet_id=df['id'])
    filtered.assign(text=df['text'])
    filtered.assign(lat=df['geo']['coordinates'][0])
    filtered.assign(lon=df['geo']['coordinates'][1])
    filtered.assign(city=df['place']['full_name'])
    filtered.assign(country_code=df['place']['country_code'])
    filtered.assign(source=df['source'])

    print(filtered.iloc[0])

    filtered.to_csv(
        "%s/extracted_2019-02-02" % project_path)

    return filtered


def text_to_words(df):
    for item in string.punctuation:
        df.str.replace(item, ' ')

    return df.str.split()


def to_corpus(df):
    dict_df = pd.DataFrame()
    dict_df.assign(tags=df['fullname']+df['country_code'])
    dict_df.assign(words=df['text'])

    print(dict_df.iloc[0])

    dict_df.to_json(
        "%s/sample_corpus_2019-02-02" % project_path,
        orient='records')

    return dict_df


def main():
    # df = extract_info("/twitter-turbo/decahose/raw/decahose.2019-02-02.p1.bz2")
    # df = extract_info("./sample_text.txt.bz2")
    df = extract_info("./baby.txt.bz2")
    to_corpus(df)


if __name__ == "__main__":
    main()
