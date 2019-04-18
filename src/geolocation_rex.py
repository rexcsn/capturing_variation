import argparse
import json
import sys
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from itertools import islice
import sklearn
from statistics import mode, median
import multiprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from copy import deepcopy
import fiona
from shapely.geometry import shape, Point

WORKING_DIR = "/shared/0/projects/location-inference/working-dir/textual_data/training/"

EARTH_RADIUS = 6371
def get_distance(lat_lon_pair1, lat_lon_pair2):
    
    dlat = np.radians(lat_lon_pair1[0]) - np.radians(lat_lon_pair2[0])
    dlon = np.radians(lat_lon_pair1[1]) - np.radians(lat_lon_pair2[1])
    a = np.square(np.sin(dlat / 2.0)) + np.cos(np.radians(lat_lon_pair2[0])) * np.cos(np.radians(lat_lon_pair1[0])) * np.square(
        np.sin(dlon / 2.0))
    great_circle_distance = 2 * np.arcsin(np.minimum(np.sqrt(a), np.repeat(1, 1)))
    d = EARTH_RADIUS * great_circle_distance

    return d


def read_in_extracted(filename):
    df = pd.DataFrame.from_csv(filename)
    return df


def build_test_dict(df):
    tweet_id_to_corpus_list = {}
    user_to_tweet_id = {}
    tweet_id_to_user = {}
    user_to_ll = {}
    for index, row in df.iterrows():
        complete_city_name = (
            row['city']+","+row['country_code']).str.replace(' ', '')
        # build tweet_id to info dict
        words_in_tweet = row['text'].str.split(r'\W+')
        tweet_id = row['tweet_id']
        tweet_id_to_corpus_list[tweet_id] = {}
        tweet_id_to_corpus_list[tweet_id]['words_in_tweet'] = words_in_tweet
        tweet_id_to_corpus_list[tweet_id]['tag'] = complete_city_name
        # idx_to_corpus_list.append(TaggedDocument(words_in_tweet, tags=complete_city_name))

        # build tweet_id to user_id and vice versa dictionary
        user_id = row['user_id']
        if user_id not in user_to_tweet_id:
            user_to_tweet_id[user_id] = []
        user_to_tweet_id[user_id].append(tweet_id)

        # set ground truth to be user's average lat lon
        if user_id not in user_to_ll:
            user_to_ll[user_id] = {}
            user_to_ll[user_id]['num_entries'] = 0
            user_to_ll[user_id]['lat'] = 0
            user_to_ll[user_id]['lon'] = 0

        # Use user's average lat lon as ground truth?
        user_to_ll[user_id]['num_entries'] += 1
        user_to_ll[user_id]['lat'] \
            = (user_to_ll[user_id]['lat']
               + float(row['lat'])) \
            / user_to_ll[user_id]['num_entries']

        user_to_ll[user_id]['lon'] \
            = (user_to_ll[user_id]['lon']
               + float(row['lon'])) \
            / user_to_ll[user_id]['num_entries']

        #
        tweet_id_to_user[tweet_id] = user_id

    return tweet_id_to_corpus_list, tweet_id_to_user, user_to_tweet_id, user_to_ll


def split_train_val_test(df, test_ratio=0.1, val_ratio=0.2):

    train_val, test_df = train_test_split(df, test_size=test_ratio)
    train_df, val_df = train_test_split(df, test_size=val_ratio)

    return train_df, val_df, test_df


def get_training_corpus_and_loc_to_ll(train_df):
    loc_to_ll = {}
    corpus_list = []
    # build loc_to_ll dict
    for index, row in train_df.iterrows():
        complete_city_name = (
            row['city']+","+row['country_code']).str.replace(' ', '')
        if complete_city_name not in loc_to_ll:
            loc_to_ll[complete_city_name] = {}
            loc_to_ll[complete_city_name]['num_entries'] = 0
            loc_to_ll[complete_city_name]['lat'] = 0
            loc_to_ll[complete_city_name]['lon'] = 0

        # Average lat lons in the same city
        loc_to_ll[complete_city_name]['num_entries'] += 1
        loc_to_ll[complete_city_name]['lat'] \
            = (loc_to_ll[complete_city_name]['lat']
               + float(row['lat'])) \
            / loc_to_ll[complete_city_name]['num_entries']

        loc_to_ll[complete_city_name]['lon'] \
            = (loc_to_ll[complete_city_name]['lon']
               + float(row['lon'])) \
            / loc_to_ll[complete_city_name]['num_entries']

        words_in_tweet = row['text'].str.split(r'\W+')
        corpus_list.append(TaggedDocument(
            words_in_tweet, tags=complete_city_name))

    return corpus_list, loc_to_ll


def train_Doc2Vec(corpus, window=15, min_occurrence=10,
                  cores=int(multiprocessing.cpu_count()/2)):

    model = Doc2Vec(size=300, window=window, min_count=min_occurrence, negative=5, hs=0,
                    workers=cores, iter=10, sample=0.00001, dm=0, dbow_words=1)

    print('\nbuilding model')
    model.build_vocab(corpus)
    print('\ntraining model')
    model.train(corpus)
    print('DONE training!')

    model.delete_temporary_training_data(
        keep_doctags_vectors=True, keep_inference=True)

    print("Saving model to %strain.model" % WORKING_DIR)
    model.save('%strained.model' % WORKING_DIR)

    return model


def infer_val(model, tweet_id_to_corpus_list, tweet_id_to_user, loc_to_ll):
    tweet_id_to_pred = {}
    user_to_pred = {}
    for tweet_id in tweet_id_to_corpus_list:
        corpus = tweet_id_to_corpus_list[tweet_id]['words_in_tweet']
        inferred_vector = model.infer_vector(corpus)
        # get the tag of the most similar vector
        tag, score = model.docvecs.most_similar([inferred_vector], topn=1)
        tweet_id_to_pred[tweet_id] = tag
        # append tag to user's prediction list
        user_id = tweet_id_to_user[tweet_id]
        if user_id not in user_to_pred:
            user_to_pred[user_id] = []
        user_to_pred[user_id].append(tag)
    
    # set the most frequent tag as user's predicted location
    for user_id in user_to_pred:
        user_to_pred[user_id] = mode(user_to_pred[user_id])
    

    return tweet_id_to_pred, user_to_pred


def evaluate_pred(loc_to_ll, user_to_pred, user_to_ll):
    error_list = []
    for user_id in user_to_pred:
        predicted_tag = user_to_pred[user_id]
        predicted_ll = loc_to_ll[predicted_tag]
        actual_ll = user_to_ll[user_id]
        error = get_distance((predicted_ll['lat'], predicted_ll['lon']), (actual_ll['lat'], actual_ll['lon']))
        error_list.append(error)

    return error


def analyze_error(pred_error):
    within = 0
    total = len(pred_error)
    for e in pred_error:
        if e < 161:
            within += 1
    
    acc_161 = within/float(total)
    median_error = median(pred_error)

    return acc_161, median_error


def main():
    # extracted contains user_id, tweet_id, text, lat, lon, city, country_code, source

    df = read_in_extracted(
        "/twitter-turbo/decahose/raw\
        /decahose.2019-02-02.p1.bz2\
        /extracted_2019-02-02")

    # get training, val, test sets
    train_df, val_df, test_df = split_train_val_test(df)

    # get training corpus, location to latlon map
    train_corpus, loc_to_ll = get_training_corpus_and_loc_to_ll(train_df)

    # build dicts for validation set
    val_tweet_id_to_corpus_list, val_tweet_id_to_user, \
        val_user_to_tweet_id, user_to_ll = build_test_dict(val_df)  # use TaggedDocument instead of dict, reuse existing corpus?

    # train model
    model = train_Doc2Vec(train_corpus)  # save model somewhere

    # infer on validation
    user_to_pred = infer_val(model, val_tweet_id_to_corpus_list, val_tweet_id_to_user, loc_to_ll)

    # Evaluate validation prediction
    pred_error = evaluate_pred(loc_to_ll, user_to_pred, user_to_ll)

    # Get accuracy @ 161km and median error
    acc_161, median_error = analyze_error(pred_error)


if __name__ == "__main__":
    main()
