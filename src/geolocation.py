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
parser.add_argument('--model', help='Doc2Vec model', default=None)
parser.add_argument('--restrict', help='restrict to a country, defined by ISO', choices={'AT', 'DE', 'CH', 'all'}, default=None, type=str)
parser.add_argument('--retrofit', help='retrofit vectors on Lameli map', choices={'nuts2', 'nuts3', 'lameli'}, default=None)
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
        # try:
        #     # if args.restrict is None or args.restrict == location['country']:
        #     if args.restrict is None or location['country'] in args.restrict:
        #         locations[location['city']] = [(location['lat'], location['lng']), location['country']]
        # except KeyError:
        #     # print('wrong input format! City key should be "city"')
        #     # sys.exit()
        #     # if args.restrict is None or args.restrict == location['country']:
        #     if args.restrict is None or location['country'] in args.restrict:
        #         locations[location['location']] = [(location['lat'], location['lng']), location['country']]

with open(args.counts, encoding='utf-8') as corpus_file:
    city_density = dict(Counter(json.load(corpus_file)).most_common(n=args.top))


print('loading model and cities', file=sys.stderr, )
model = Doc2Vec.load(args.model)
eligible_cities = [city for city in locations if city in city_density if city_density[city] >= args.min and city_density[city] <= args.max and city in model.docvecs]
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

word_matrix = np.array([model[word] for word in model.wv.index2word])
w2i = {w: i for i, w in enumerate(model.wv.index2word)}

if args.retrofit is not None:
    vectors = np.array([model.docvecs[city] for city in eligible_cities])
    retrofitting_region_outlines = {}
    retro_shape_file = '/Users/dirkhovy/Dropbox/working/lowlands/GeoStats/data/nuts/NUTS_RG_03M_2010.shp'
    print("reading region outline from %s" % retro_shape_file, end=' ', file=sys.stderr)
    retro_fiona_shapes = fiona.open(retro_shape_file)

    for item in islice(retro_fiona_shapes, None):
        level = int(item['properties']['STAT_LEVL_'])
        if (level == 2 and args.retrofit == 'nuts2') or (level == 3 and args.retrofit == 'nuts3'):
            nuts_id = item['properties']['NUTS_ID']
            retrofitting_region_outlines[nuts_id] = shape(item['geometry'])

    print("done\n", file=sys.stderr)
    # get the dictionary
    print("computing region dictionary", end=' ', file=sys.stderr)
    retro_regions2cities = defaultdict(set)
    for c, city in enumerate(eligible_cities):
        # if locations[city][-2] != "DE":
        #     continue
        city_location = locations[city][-1]
        for region, shape_ in retrofitting_region_outlines.items():
            if shape_.contains(city_location):
                retro_regions2cities[region].add(c)
                break

    # map from city ID to city IDs in the same region
    print("and neighbor dictionary", file=sys.stderr)
    retro_neighbors = {}
    for region, cities in retro_regions2cities.items():
        for city in cities:
            retro_neighbors[city] = list(cities.copy())
            # remove city ID from its own neighbors
            retro_neighbors[city].remove(city)

    new_vectors = deepcopy(vectors)
    # normalize new vectors?
    # for c in range(len(eligible_cities)):
    #     vectors[c] /= np.sqrt((vectors[c]**2).sum() + 1e-6)

    # run retrofitting
    print("retrofitting", file=sys.stderr)
    num_iters = 10
    for it in range(num_iters):
        print("\t{}:".format(it + 1), end=' ', file=sys.stderr)
        # loop through every city
        for c in range(len(eligible_cities)):
            print(".", end='', file=sys.stderr)
            city_neighbours = retro_neighbors[c]
            num_neighbours = len(city_neighbours)
            # no neighbours, pass - use data estimate
            if num_neighbours == 0:
                continue

            # the weight of the data estimate is the number of neighbours, plus the sum of all neighboring vectors, normalized
            new_vectors[c] = ((num_neighbours * vectors[c]) + new_vectors[city_neighbours].sum(axis=0)) / (2 * num_neighbours)

        print('', file=sys.stderr)

    print('\nfinding translation matrix', file=sys.stderr)
    translation_matrix = np.linalg.lstsq(vectors, new_vectors)[0]
    print(translation_matrix.shape, file=sys.stderr)
    word_matrix = np.dot(word_matrix, translation_matrix)

X = []
y = []
train_texts = []

print('reading training', file=sys.stderr, )
counter = 0
for line in open(args.train_corpus):
    line = json.loads(line)
    if line['tags'][0] in targets and len(line['words']) >= args.min_length:
        # TODO: replace vector by avg or sum of word vectors
        # X.append(word_matrix[[w2i[word] for word in line['words'] if word in w2i]].sum(axis=0))
        X.append(model.infer_vector(line['words'], steps=500))
        train_texts.append(' '.join([word for word in line['words'] if word in model.wv.vocab]))
        y.append(targets[line['tags'][0]])
        counter += 1
        if counter%1000==0:
            print(counter, file=sys.stderr, flush=True)
        elif counter%50==0:
            print('.', file=sys.stderr, end='', flush=True)

    if len(X) == args.train_limit:
        break

X1 = []
y1 = []
test_texts = []
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
            # TODO: replace vector by avg or sum of word vectors
            # X1.append(word_matrix[[w2i[word] for word in line['words'] if word in w2i]].sum(axis=0))
            X1.append(model.infer_vector(line['words'], steps=500))
            test_texts.append(' '.join([word for word in line['words'] if word in model.wv.vocab]))
            y1.append(city_id)
            test_samples[city_id] += 1

            counter += 1
            if counter%1000==0:
                print(counter, file=sys.stderr, flush=True)
            elif counter%50==0:
                print('.', file=sys.stderr, end='', flush=True)

    if len(X1) == args.test_limit:
        break

# print(test_proportions, '\n', Counter(y1))


print('\n\nfitting model on %s instances' % (len(y)), file=sys.stderr, )
clf = LogisticRegression(n_jobs=-1)
clf.fit(X, y)
predictions = clf.predict(X1)
gold = y1

distances = np.array([P[inverse_targets[gold[i]]][inverse_targets[predictions[i]]] for i in range(len(predictions))])
acc100 = np.array([P100[inverse_targets[gold[i]]][inverse_targets[predictions[i]]] for i in range(len(predictions))])

# TODO: implement random
# random =
clf2 = LogisticRegression(n_jobs=-1)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_texts)

# TODO: test without tfidf
tf_transformer = TfidfTransformer()
X_train_tfidf = tf_transformer.fit_transform(X_train_counts)

X_test_counts = count_vect.transform(test_texts)
X_test_tfidf = tf_transformer.transform(X_test_counts)

clf2.fit(X_train_tfidf, y)
predictions2 = clf2.predict(X_test_tfidf)

distances2 = np.array([P[inverse_targets[gold[i]]][inverse_targets[predictions2[i]]] for i in range(len(predictions2))])
acc1002 = np.array([P100[inverse_targets[gold[i]]][inverse_targets[predictions2[i]]] for i in range(len(predictions2))])


print("#cities\t&\tprecision\t&\tprecision@100\t&\tmean dist.")
print("BOW\t%s\t&\t%.2f\t&\t%.2f\t&\t%.2f" % (len(eligible_cities), precision_score(gold, predictions2, average='micro'), acc1002.mean(), np.median(distances2)))
print("D2V\t%s\t&\t%.2f\t&\t%.2f\t&\t%.2f" % (len(eligible_cities), precision_score(gold, predictions, average='micro'), acc100.mean(), np.median(distances)))
