import os
from itertools import islice
import numpy as np
from collections import defaultdict, Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import completeness_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import v_measure_score
from copy import deepcopy

try:
    r = os.system('python -c "import matplotlib.pyplot as plt;plt.figure()"')
    if r == 256:
        import matplotlib
        matplotlib.use('Agg')
    else:
        import matplotlib
        matplotlib.use('Qt5Agg')
except RuntimeError:
    pass
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Doc2Vec
from matplotlib import colors
from mpl_toolkits.basemap import Basemap
from numpy import cos, radians
from sklearn.cluster import AgglomerativeClustering, KMeans

import fiona
from shapely.geometry import shape, Point

import seaborn
seaborn.set('paper')

cm = plt.get_cmap('gist_rainbow')
EARTH_RADIUS = 6371

country_boxes = {
    'germany': [47., 55.1, 5.5, 15.5, 50., 10., 10.]
}

parser = argparse.ArgumentParser('put things on a map')
parser.add_argument('--affinity', help='cluster affinity', choices={'l1', 'l2', 'euclidean', 'manhattan', 'cosine'}, default='euclidean')
parser.add_argument('--alpha', help='retrofitting trade-off parameter. alpha=1 reproduces the input', default=0.5, type=float)
parser.add_argument('--clusters', help='number of clusters', default=None, type=int)
parser.add_argument('--country', help='map restriction', choices=country_boxes.keys(), default='germany')
parser.add_argument('--csv', help='output CSV of clustering', default=None)
parser.add_argument('--locations', help='location files', nargs='+')
parser.add_argument('--linkage', help='cluster linkage', choices={'complete', 'average', 'ward'}, default='ward')
parser.add_argument('--max', help='maximum number of conversations required of city', default=99999999999, type=int)
parser.add_argument('--min', help='minimum number of conversations required of city', default=200, type=int)
parser.add_argument('--model', help='Doc2Vec model', default=None)
parser.add_argument('--prefix', help='prefix for city counts and corpus')
parser.add_argument('--resolution', help='map resolution', choices={'l', 'i', 'h', 'f'}, default='l', type=str)
parser.add_argument('--restrict', help='restrict to a country, defined by ISO', choices={'DE'}, default='DE', type=str)
parser.add_argument('--retrofit', help='retrofit vectors on Lameli map', choices={'nuts2', 'nuts3', 'lameli'}, default=None)
parser.add_argument('--save', help='save as PNG file', default=None)
parser.add_argument('--show_cities', help='show city names on map', action='store_true')
parser.add_argument('--show_nuts', help='show NUTS oracle scores', action='store_true')
parser.add_argument('--vectors', help='vectors to use', choices={'d2v', 'bow'}, default='d2v', type=str)

args = parser.parse_args()

KMEANS_AVG = 5

country = args.country

width = country_boxes[country][3] - country_boxes[country][2]
height = country_boxes[country][1] - country_boxes[country][0]
fig = plt.figure(figsize=(width, height))
ax = fig.add_subplot(111)

m = Basemap(llcrnrlat=country_boxes[country][0],
            urcrnrlat=country_boxes[country][1],
            llcrnrlon=country_boxes[country][2],
            urcrnrlon=country_boxes[country][3],
            resolution=args.resolution ,projection='gnom',
            lat_0=country_boxes[country][4],
            lon_0=country_boxes[country][5],
            lat_ts=country_boxes[country][6])

def get_shortest_in(needle, haystack):
    '''

    :param needle: single (lat,long) tuple.
    :param haystack: numpy array to find the point in that has the shortest distance to needle
    :return:
    '''
    dlat = radians(haystack[0]) - radians(needle[0])
    dlon = radians(haystack[1]) - radians(needle[1])
    a = np.square(np.sin(dlat / 2.0)) + cos(radians(needle[0])) * np.cos(radians(haystack[0])) * np.square(
        np.sin(dlon / 2.0))
    great_circle_distance = 2 * np.arcsin(np.minimum(np.sqrt(a), np.repeat(1, 1)))
    d = EARTH_RADIUS * great_circle_distance
    return d.tolist()[0]


# read in locations
locations = {}
for location_file in args.locations:
    location_list = json.load(open(location_file))
    for location in location_list:
        try:
            if args.restrict is None or args.restrict == location['country']:
                locations[location['city']] = [(location['lat'], location['lng']), location['country'], Point(location['lng'], location['lat'])]
        except KeyError:
            # print('wrong input format! City key should be "city"')
            # sys.exit()
            if args.restrict is None or args.restrict == location['country']:
                locations[location['location']] = [(location['lat'], location['lng']), location['country']]

with open('%s.citycounts' % args.prefix, encoding='utf-8') as corpus_file:
    city_density = json.load(corpus_file)

lower_bound = min(city_density.values())
upper_bound = max(city_density.values())
divisor = upper_bound - lower_bound

# scale counts
city_density_scaled = {}
for city in city_density:
    city_density_scaled[city] = ((city_density[city] - lower_bound) / divisor) * 500 + 50

model = Doc2Vec.load(args.model)
eligible_cities = [city for city in locations if city in city_density if city_density[city] >= args.min and city_density[city] <= args.max and city in model.docvecs]

print('\n\tusing %s cities' % len(eligible_cities), file=sys.stderr)

color_names = ['red' for city in eligible_cities]
cMap = 'Reds'

proximity = np.zeros((len(eligible_cities), len(eligible_cities)))
for i, city1 in enumerate(eligible_cities):
    for j in range(i+1, len(eligible_cities)):
        city2 = eligible_cities[j]
        distance = get_shortest_in(locations[city1][0], locations[city2][0])

        proximity[i, j] = distance
        proximity[j, i] = distance

# invert distances
furthest = proximity.max()
proximity = furthest - proximity

cluster_colors = [cm(1. * i / args.clusters) for i in range(args.clusters)]


# get BOW representations
if args.vectors == 'bow':
    try:
        X_train_tfidf = np.load('%s-BOW.npz' % args.prefix)['bow']
        print('loaded training', file=sys.stderr, )
    except:
        city_word_counts = defaultdict(Counter)
        print('reading training', file=sys.stderr, )
        for line in open('%s.corpus' % args.prefix):
            line = json.loads(line)
            city = line['tags'][0]
            if city in eligible_cities:
                city_word_counts[city].update(line['words'])
        print('done', file=sys.stderr, )

        # process
        count_vect = DictVectorizer()
        X_train_counts = count_vect.fit_transform([city_word_counts[city] for city in eligible_cities])
        tf_transformer = TfidfTransformer()
        X_train_tfidf = tf_transformer.fit_transform(X_train_counts).todense()
        np.savez('%s-BOW.npz' % args.prefix, bow=X_train_tfidf)
    print('{} BOW features'.format(X_train_tfidf.shape[1]), file=sys.stderr, )

    # decide whether to use D2V or BOW
    vectors = X_train_tfidf
else:
    # get city representations
    d2v_vectors = np.array([model.docvecs[city] for city in eligible_cities])
    vectors = d2v_vectors

shape_file = '/Users/dirkhovy/Dropbox/working/sociolinguistics/playground/Lameli maps/Lameli.shp'
print("reading region outline from %s" % shape_file, end=' ', file=sys.stderr)
fiona_shapes = fiona.open(shape_file)
lameli_region_outlines = {}
for item in islice(fiona_shapes, None):
    lameli_region_outlines[item['properties']['id']] = shape(item['geometry'])


if args.retrofit is not None:
    alpha = args.alpha
    beta = 1 - args.alpha
    retrofitting_region_outlines = {}
    if args.retrofit != 'lameli':
        retro_shape_file = '/Users/dirkhovy/Dropbox/working/lowlands/GeoStats/data/nuts/NUTS_RG_03M_2010.shp'
        print("reading region outline from %s" % retro_shape_file, end=' ', file=sys.stderr)
        retro_fiona_shapes = fiona.open(retro_shape_file)

        for item in islice(retro_fiona_shapes, None):
            level = int(item['properties']['STAT_LEVL_'])
            if (level == 2 and args.retrofit == 'nuts2') or (level == 3 and args.retrofit == 'nuts3'):
                nuts_id = item['properties']['NUTS_ID']
                retrofitting_region_outlines[nuts_id] = shape(item['geometry'])
    else:
        retrofitting_region_outlines = lameli_region_outlines.copy()

    print("done\n", file=sys.stderr)
    # get the dictionary
    print("computing region dictionary", end=' ', file=sys.stderr)
    retro_regions2cities = defaultdict(set)
    for c, city in enumerate(eligible_cities):
        if locations[city][-2] != "DE":
            continue
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
    print("retrofitting with alpha={}".format(alpha), file=sys.stderr)
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
            new_vectors[c] = (alpha * (num_neighbours * vectors[c]) + beta * new_vectors[city_neighbours].sum(axis=0)) / (alpha * num_neighbours + beta * num_neighbours)

        print('', file=sys.stderr)

    vectors = new_vectors
    # # cluster the new vectors
    # print("cluster retrofitted data", file=sys.stderr)
    # retro_clustering_algo = AgglomerativeClustering(n_clusters=args.clusters, connectivity=proximity)
    # retro_cluster_ids = clustering_algo.fit_predict(X=new_vectors)

# do agglomerative clustering with structure
print('agglomerative clustering', file=sys.stderr, )
clustering_algo = AgglomerativeClustering(n_clusters=args.clusters, connectivity=proximity, affinity=args.affinity, linkage=args.linkage)
cluster_ids = clustering_algo.fit_predict(X=vectors)
color_names = [cluster_colors[c] for i, c in enumerate(cluster_ids) if locations[eligible_cities[i]][-2] == "DE"]
cMap = colors.ListedColormap(cluster_colors)
print('done', file=sys.stderr, )

# do kmeans clustering
print('kmeans clustering', file=sys.stderr, )
dumb_cluster_ids = []
for x in range(KMEANS_AVG):
    dumb_cluster = KMeans(n_jobs=-1, n_clusters=args.clusters)
    dumb_cluster_ids.append(dumb_cluster.fit_predict(vectors))
    # dumb_cluster_ids = dumb_cluster.fit_predict(X_train_tfidf)
print('done', file=sys.stderr, )


if args.show_nuts:
    NUTS_shape_file = '/Users/dirkhovy/Dropbox/working/lowlands/GeoStats/data/nuts/NUTS_RG_03M_2010.shp'
    print("reading country outline from %s" % NUTS_shape_file, end=' ', file=sys.stderr)
    NUTS_shapes = fiona.open(NUTS_shape_file)

    NUTS2_outlines = {}
    NUTS3_outlines = {}
    for item in islice(NUTS_shapes, None):
        nuts_id = None
        if item['properties']['STAT_LEVL_'] == 2:
            NUTS2_outlines[item['properties']['NUTS_ID']] = shape(item['geometry'])
        elif item['properties']['STAT_LEVL_'] == 3:
            NUTS3_outlines[item['properties']['NUTS_ID']] = shape(item['geometry'])
    print("done\n", file=sys.stderr)


gold = []
pred = []
dumb_pred = np.zeros((KMEANS_AVG, len(eligible_cities)))

oracle2 = []
oracle3 = []
retro_pred = []

for c, city in enumerate(eligible_cities):
    if locations[city][-2] != "DE":
        continue
    city_location = locations[city][-1]
    solution = '9999'
    for region, shape_ in lameli_region_outlines.items():
        if shape_.contains(city_location):
            solution = region
            break
    if solution == '9999':
        print('%s is in %s' % (city, solution), file=sys.stderr)

    if args.show_nuts:
        for NUTS2_region, shape_ in NUTS2_outlines.items():
            if shape_.contains(city_location):
                NUTS2_solution = NUTS2_region
                break
        for NUTS3_region, shape_ in NUTS3_outlines.items():
            if shape_.contains(city_location):
                NUTS3_solution = NUTS3_region
                break

        oracle2.append(NUTS2_solution)
        oracle3.append(NUTS3_solution)

    # if args.retrofit:
    #     retro_pred.append(retro_cluster_ids[c])

    gold.append(solution)
    pred.append(cluster_ids[c])
    for i in range(KMEANS_AVG):
        dumb_pred[i][c] = dumb_cluster_ids[i][c]

dumb_pred_v = np.array([v_measure_score(gold, dumb_pred[i, :]) for i in range(KMEANS_AVG)]).mean()
dumb_pred_h = np.array([homogeneity_score(gold, dumb_pred[i, :]) for i in range(KMEANS_AVG)]).mean()
dumb_pred_c = np.array([completeness_score(gold, dumb_pred[i, :]) for i in range(KMEANS_AVG)]).mean()

# if args.retrofit:
#     retro_pred_v = v_measure_score(gold, retro_pred)
#     retro_pred_h = homogeneity_score(gold, retro_pred)
#     retro_pred_c = completeness_score(gold, retro_pred)
#     print(retro_pred_v, retro_pred_h, retro_pred_c)

# print('clusters\tV-measure\thomogeneity\tcompleteness')
print('%s\t&\t%.2f\t&\t%.2f\t&\t%.2f\t&\t%.2f\t&\t%.2f\t&\t%.2f' % (args.clusters, v_measure_score(gold, pred), homogeneity_score(gold, pred), completeness_score(gold, pred), dumb_pred_v, dumb_pred_h, dumb_pred_c))
# print('%.2f\t&\t%.2f\t&\t%.2f\t&\t%.2f\t&\t%.2f\t&\t%.2f' % (v_measure_score(gold, oracle2), homogeneity_score(gold, oracle2), completeness_score(gold, oracle2), v_measure_score(gold, oracle3), homogeneity_score(gold, oracle3), completeness_score(gold, oracle3)))


# m.readshapefile('/Users/dirkhovy/Dropbox/working/lowlands/sociolinguistics/playground/Lameli maps/Lameli', 'de', drawbounds=True)
#
# x, y, z = zip(*[(locations[city][0][1], locations[city][0][0], city_density_scaled[city]) for city in eligible_cities if locations[city][-2] == "DE"])
# # transform coordinates into map space
# x1, y1 = m(x, y)
# m.scatter(x1, y1, z, marker='o', c=color_names, cmap=cMap, zorder=4, alpha=0.8)
#
#
# if args.show_cities:
#     # add city names
#     lons, lats= zip(*[coords for city, (coords, iso) in locations.items() if city in eligible_cities and locations[city][-2] == "DE"])
#     lats1, lons1 = m(list(lats), list(lons))
#
#     for name, xpt, ypt in zip(eligible_cities, lats1, lons1):
#         if  city_density_scaled[name] >= 250 or name in {'Basel', 'Zürich', 'Bern', 'Genf', 'Trier', 'Saarbrücken', 'Lausanne'}:
#             plt.text(xpt, ypt, name, fontsize="12", zorder=5)
#
# patches = []
# for info, dshape in zip(m.de_info, m.de):
#     ax.add_patch(Polygon(np.array(shape_, dtype=np.float), True))
#     # patches.append(Polygon(np.array(shape_), True))
# p = PatchCollection(patches, alpha=0.5,  zorder=3, facecolors='grey', cmap='rainbow')
# ax.add_collection(p)
#
# if args.save:
#     print("Saving to '%s'" % (args.save), file=sys.stderr, flush=True)
#     fig.savefig(args.save, dpi=300, bbox_inches='tight', pad_inches=0.1)
#     print("done", file=sys.stderr, flush=True)
# else:
#     fig.tight_layout()
#     plt.show()
