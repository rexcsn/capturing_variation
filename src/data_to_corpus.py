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

project_path = "/shared/0/projects/location-inference/working-dir/textual_data/"
# project_path = "."


def extract_info(filename, name):
    labels = ['user_id', 'tweet_id', 'text', 'lat', 'lon', 'city', 'country_code', 'source']
    file = bz2.BZ2File(filename)
    data = {}
    for item in labels:
        data[item] = []
    for line in file:
        js = json.loads(line)
        try:
            if js['user']['id'] != "null" and js['id'] != "null" \
                    and js['text'] != "null" and js['geo']['coordinates'][0] != "null" \
                    and js['geo']['coordinates'][1] != "null" and js['place']['full_name'] != "null" \
                    and js['place']['country_code'] != "null" and js['source'] != "null":

                data['user_id'].append(js['user']['id'])
                data['tweet_id'].append(js['id'])
                data['text'].append(js['text'])
                data['lat'].append(js['geo']['coordinates'][0])
                data['lon'].append(js['geo']['coordinates'][1])
                data['city'].append(js['place']['full_name'])
                data['country_code'].append(js['place']['country_code'])
                data['source'].append(js['source'])

        except Exception:
            continue

    filtered = pd.DataFrame(data)
    # df = df[labels]
    #
    # filtered = pd.DataFrame()
    #
    # filtered = filtered.assign(user_id=df['user'].apply(pd.Series)['id'],
    #                            tweet_id=df['id'],
    #                            text=df['text'],
    #                            lat=df['geo'].apply(pd.Series)['coordinates'][0],
    #                            lon=df['geo'].apply(pd.Series)['coordinates'][1],
    #                            city=df['place'].apply(pd.Series)['full_name'],
    #                            country_code=df['place'].apply(pd.Series)['country_code'],
    #                            source=df['source']
    #                            )

    filtered.to_csv(
        "%sextracted_%s" % (project_path, name))
    print("Saved df to %sextracted_%s !!!" % (project_path, name))

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
        orient='records', lines=True)
    print("Saved corpus to %s/sample_corpus_2019-02-02 !!!" % project_path)

    return dict_df


def main():
    for i in range(2, 9):
        df = extract_info("/twitter-turbo/decahose/raw/decahose.2019-02-0%s.p1.bz2" % i, "2019-02-0%s.p1")
        df = extract_info("/twitter-turbo/decahose/raw/decahose.2019-02-0%s.p2.bz2" % i, "2019-02-0%s.p2")
    # df = extract_info("./sample_text.txt.bz2")
    # df = extract_info("./baby.txt.bz2")
    # to_corpus(df)


if __name__ == "__main__":
    main()
