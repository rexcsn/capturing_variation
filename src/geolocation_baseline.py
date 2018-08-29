import argparse
import json
import sys
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from itertools import islice
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from copy import deepcopy
import fiona
from shapely.geometry import shape, Point


parser = argparse.ArgumentParser('evaluate geolocation')
parser.add_argument('--counts', help='city counts')
parser.add_argument('--train_corpus', help='training corpus')
parser.add_argument('--train_limit', help='max number of training instances', type=int, default=20000)
parser.add_argument('--test_corpus', help='test corpus')
parser.add_argument('--test_limit', help='max number of testing instances', type=int, default=5000)
parser.add_argument('--locations', help='location files', nargs='+')
parser.add_argument('--max', help='maximum number of conversations required of city', default=99999999999, type=int)
parser.add_argument('--min', help='minimum number of conversations required of city', default=200, type=int)
parser.add_argument('--min_length', help='minimum number of words per conversations required', default=100, type=int)
parser.add_argument('--restrict', help='restrict to a country, defined by ISO', choices={'AT', 'DE', 'CH', 'all'}, default=None, type=str)
parser.add_argument('--top', help='number of cities', default=100, type=int)

args = parser.parse_args()

restrictions = {'AT': {'AT'}, 'DE': {'DE'}, 'CH': {'CH'}, 'all': {'AT', 'DE', 'CH'}, None: None}
args.restrict = restrictions[args.restrict]

EARTH_RADIUS = 6371
def get_shortest_in(needle, haystack):
    '''
    :param needle: single (lat,long) tuple.
    :param haystack: numpy array to find the point in that has the shortest distance to needle
    :return:
    '''
    dlat = np.radians(haystack[0]) - np.radians(needle[0])
    dlon = np.radians(haystack[1]) - np.radians(needle[1])
    a = np.square(np.sin(dlat / 2.0)) + np.cos(np.radians(needle[0])) * np.cos(np.radians(haystack[0])) * np.square(
        np.sin(dlon / 2.0))
    great_circle_distance = 2 * np.arcsin(np.minimum(np.sqrt(a), np.repeat(1, 1)))
    d = EARTH_RADIUS * great_circle_distance
    return d.tolist()[0]


# read in locations
print('reading locations', file=sys.stderr, )
locations = {}
for location_file in args.locations:
    location_list = json.load(open(location_file))
    for location in location_list:
        try:
            if args.restrict is None or location['country'] in args.restrict:
                locations[location['city']] = [(location['lat'], location['lng']), location['country'], Point(location['lng'], location['lat'])]
        except KeyError:
            # print('wrong input format! City key should be "city"')
            # sys.exit()
            if args.restrict is None or location['country'] in args.restrict:
                locations[location['location']] = [(location['lat'], location['lng']), location['country']]

# get the top N cities (by number of counts)
with open(args.counts, encoding='utf-8') as corpus_file:
    city_density = dict(Counter(json.load(corpus_file)).most_common(n=args.top))


eligible_cities = [city for city in locations if city in city_density if city_density[city] >= args.min and city_density[city] <= args.max]
targets = {city: c for c, city in enumerate(eligible_cities)}
inverse_targets = {c: city for city, c in targets.items()}
print('using %s cities' % len(eligible_cities), file=sys.stderr, )


# get distances
print('computing distances', file=sys.stderr, )
proximity = np.zeros((len(eligible_cities), len(eligible_cities)))
for i, city1 in enumerate(eligible_cities):
    for j in range(i+1, len(eligible_cities)):
        city2 = eligible_cities[j]
        distance = get_shortest_in(locations[city1][0], locations[city2][0])

        proximity[i, j] = distance
        proximity[j, i] = distance
P = pd.DataFrame(data=proximity, columns=eligible_cities, index=eligible_cities)
P100 = pd.DataFrame(data=np.where(proximity < 161, 1, 0), columns=eligible_cities, index=eligible_cities)

y = []

print('reading training', file=sys.stderr, )
counter = 0
for line in open(args.train_corpus):
    line = json.loads(line)
    if line['tags'][0] in targets and len(line['words']) >= args.min_length:
        y.append(targets[line['tags'][0]])
        counter += 1
        if counter%1000==0:
            print(counter, file=sys.stderr, flush=True)
        elif counter%50==0:
            print('.', file=sys.stderr, end='', flush=True)

    if len(y) == args.train_limit:
        break

y1 = []
test_proportions = {city: count/args.train_limit*args.test_limit for city, count in Counter(y).items()}
test_samples = defaultdict(int)

print('\n\nreading test', file=sys.stderr, )
counter = 0
for line in open(args.test_corpus):
    line = json.loads(line)
    if line['tags'][0] in targets and len(line['words']) >= args.min_length:
        # sample:
        city_id = targets[line['tags'][0]]
        if city_id in test_proportions and test_samples[city_id] < test_proportions[city_id]:
            y1.append(city_id)
            test_samples[city_id] += 1

            counter += 1
            if counter%1000==0:
                print(counter, file=sys.stderr, flush=True)
            elif counter%50==0:
                print('.', file=sys.stderr, end='', flush=True)

    if len(y1) == args.test_limit:
        break


predictions = [Counter(y).most_common(1)[0][0]] * len(y1)
gold = y1

distances = np.array([P[inverse_targets[gold[i]]][inverse_targets[predictions[i]]] for i in range(len(predictions))])
acc100 = np.array([P100[inverse_targets[gold[i]]][inverse_targets[predictions[i]]] for i in range(len(predictions))])


print("#cities\t&\tprecision\t&\tprecision@100\t&\tmean dist.")
print("most frequent\t%s\t&\t%.2f\t&\t%.2f\t&\t%.2f" % (len(eligible_cities), precision_score(gold, predictions, average='micro'), acc100.mean(), np.median(distances)))
