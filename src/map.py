import os
from copy import deepcopy
from itertools import islice

import fiona
from shapely.geometry import shape, Point

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
from collections import Counter, defaultdict
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Doc2Vec
from matplotlib import colors
from mpl_toolkits.basemap import Basemap
from numpy import cos, radians
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import NMF
# from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import seaborn
seaborn.set('paper')

cm = plt.get_cmap('gist_rainbow')
EARTH_RADIUS = 6371

country_boxes = {
    'germany': [47., 55.1, 5.5, 15.5, 50., 10., 10.],
    'german-speaking': [45.818, 55.1, 5.5, 17.2459, 50., 10., 10.],
    'austria': [46.3108, 49.0697, 9.451, 17.2459, 47.5, 13., 13.],
    'switzerland': [45.818, 47.85, 5.8887, 10.5579, 46., 7., 7.],
    'denmark': [54.5, 58., 7.5, 15.5, 56., 11.5, 11.5]
}

parser = argparse.ArgumentParser('put things on a map')
parser.add_argument('--clusters', help='number of clusters', default=None, type=int)
parser.add_argument('--corpus', help='city counts')
parser.add_argument('--country', help='map area restriction', choices=country_boxes.keys(), default='german-speaking')
parser.add_argument('--csv', help='output CSV of clustering', default=None)
parser.add_argument('--dendrogram', help='show dendrogram', action='store_true')
parser.add_argument('--locations', help='location files', nargs='+')
parser.add_argument('--max', help='maximum number of conversations required of city', default=99999999999, type=int)
parser.add_argument('--min', help='minimum number of conversations required of city', default=200, type=int)
parser.add_argument('--model', help='Doc2Vec model', default=None)
parser.add_argument('--pca', help='use PCA decomposition to RGB channels', action='store_true')
parser.add_argument('--prototypes', help='find N prototypes for all cities in a cluster (requires clustering first, duh...)', default=None, type=int)
parser.add_argument('--resolution', help='map resolution', choices={'l', 'i', 'h', 'f'}, default='l', type=str)
parser.add_argument('--restrict', help='country restriction', choices={'AT', 'DE', 'CH', 'all'}, default=None)
parser.add_argument('--retrofit', help='retrofit vectors on Lameli map', choices={'nuts2', 'nuts3', 'lameli'}, default=None)
parser.add_argument('--save', help='save as PNG file', default=None)
parser.add_argument('--show_cities', help='show city names on map', action='store_true')

args = parser.parse_args()

restrictions = {'AT': {'AT'}, 'DE': {'DE'}, 'CH': {'CH'}, 'all': {'AT', 'DE', 'CH'}, None: None}
args.restrict = restrictions[args.restrict]


def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


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
            if args.restrict is None or location['country'] in args.restrict:
                locations[location['city']] = [(location['lat'], location['lng']), location['country'], Point(location['lng'], location['lat'])]
        except KeyError:
            # print('wrong input format! City key should be "city"')
            # sys.exit()
            if args.restrict is None or location['country'] in args.restrict:
                locations[location['location']] = [(location['lat'], location['lng']), location['country']]

with open(args.corpus, encoding='utf-8') as corpus_file:
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
vectors = np.array([model.docvecs[city] for city in eligible_cities])

if args.retrofit is not None:
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
        retro_shape_file = '/Users/dirkhovy/Dropbox/working/sociolinguistics/playground/Lameli maps/Lameli.shp'
        print("reading region outline from %s" % retro_shape_file, end=' ', file=sys.stderr)
        fiona_shapes = fiona.open(retro_shape_file)
        for item in islice(fiona_shapes, None):
            retrofitting_region_outlines[item['properties']['id']] = shape(item['geometry'])

    print("done\n", file=sys.stderr)
    # get the dictionary
    print("computing region dictionary", end=' ', file=sys.stderr)
    retro_regions2cities = defaultdict(set)
    for c, city in enumerate(eligible_cities):
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

    vectors = new_vectors
    # # cluster the new vectors
    # print("cluster retrofitted data", file=sys.stderr)
    # retro_clustering_algo = AgglomerativeClustering(n_clusters=args.clusters, connectivity=proximity)
    # retro_cluster_ids = clustering_algo.fit_predict(X=new_vectors)

print('\n\tusing %s cities' % len(eligible_cities))

color_names = ['red' for city in eligible_cities]
cMap = 'Reds'

# get clusters
if args.clusters:
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

    clustering_algo = AgglomerativeClustering(n_clusters=args.clusters, connectivity=proximity)
    cluster_ids = clustering_algo.fit_predict(X=vectors)
    color_names = [cluster_colors[c] for c in cluster_ids]
    cMap = colors.ListedColormap(cluster_colors)

    if args.dendrogram:
        plt.title('Hierarchical Clustering Dendrogram')
        plot_dendrogram(clustering_algo, labels=clustering_algo.labels_)
        plt.show()

    if args.prototypes is not None:
        city2cluster = {city: cid for city, cid in zip(eligible_cities, cluster_ids)}
        cluster2cities = {cid: [city for city, cid_ in city2cluster.items() if cid_ == cid] for cid in range(args.clusters)}

        for cluster_id, cities in cluster2cities.items():
            prototypes = Counter()
            word_counts = Counter()
            for city in cities:
                topN = dict(model.most_similar(model.docvecs[[city]], topn=args.prototypes))
                prototypes.update(topN)
                word_counts.update(topN.keys())

            centroid = model.docvecs[cities].mean(axis=0)

            print('CLUSTER %s\t%s' % (cluster_id + 1, ', '.join(cities)))
            best_prototypes_avg = sorted([(key, value/word_counts[key]) for key, value in prototypes.most_common(args.prototypes)], key=lambda x: x[1], reverse=True)
            print('MOST FREQUENT %s WORDS:' % args.prototypes)
            print('\t%s' % (word_counts.most_common(args.prototypes)))
            print('TOP %s WORDS AVERAGED:' % args.prototypes)
            print('\t%s' % (best_prototypes_avg))
            best_prototypes_sum = sorted([(key, value) for key, value in prototypes.most_common(args.prototypes)], key=lambda x: x[1], reverse=True)
            print('TOP %s WORDS SUMMED:' % args.prototypes)
            print('\t%s' % (best_prototypes_sum))
            print('MOST SIMILAR TO CENTROID:')
            print('\t%s' % model.most_similar([centroid], topn=args.prototypes))
            print()

    if args.csv is not None:
        with open(args.csv, 'w')as csv_file:
            csv_file.write('lat\tlng\tcluster\tname\tiso2\tnum_posts\n')
            for city, cid in sorted(zip(eligible_cities, cluster_ids)):
                (lat, lng), iso2 = locations[city]
                csv_file.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (lat, lng, cid, city, iso2, city_density[city]))


elif args.pca:
    # vectors = np.array([model.docvecs[city] for city in eligible_cities])
    # subtract negatives to guarantee positive numbers
    vectors -= vectors.min()
    pca = NMF(n_components=3, init='nndsvd', shuffle=True)
    rgb = pca.fit_transform(vectors)

    # scale values in RGB to [0-1]
    for component in range(3):
        lower_bound = np.percentile(rgb[:, component], 1, axis=0)
        upper_bound = np.percentile(rgb[:, component], 90, axis=0)
        rgb[:, component] -= lower_bound
        rgb[:, component] /= upper_bound - lower_bound
    color_names = np.clip(rgb, 0.0, 1.0)
    cMap = 'Reds'

country = args.country

width = country_boxes[country][3] - country_boxes[country][2]
height = country_boxes[country][1] - country_boxes[country][0]
fig = plt.figure(figsize=(width, height))
ax = fig.add_subplot(111)

m = Basemap(llcrnrlat=country_boxes[country][0],
            urcrnrlat=country_boxes[country][1],
            llcrnrlon=country_boxes[country][2],
            urcrnrlon=country_boxes[country][3],
            resolution=args.resolution ,projection='merc',
            lat_0=country_boxes[country][4],
            lon_0=country_boxes[country][5],
            lat_ts=country_boxes[country][6])
            #, epsg=5520)

# decorate map
m.drawcoastlines()
m.drawcountries()
# z-order guarantees that the filling does not cover the scatter plot
m.shadedrelief()
# m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 1500, verbose= True)

x, y, z = zip(*[(locations[city][0][1], locations[city][0][0], city_density_scaled[city]) for city in eligible_cities])

# transform coordinates into map space
x1, y1 = m(x, y)
m.scatter(x1, y1, z, marker='o', c=color_names, cmap=cMap, zorder=4, alpha=0.8, edgecolor='k')

if args.show_cities:
    # add city names
    lons, lats= zip(*[coords for city, (coords, iso, _) in locations.items() if city in eligible_cities])
    lats1, lons1 = m(list(lats), list(lons))

    for name, xpt, ypt in zip(eligible_cities, lats1, lons1):
        if  city_density_scaled[name] >= 250 or name in {'Basel', 'Zürich', 'Bern', 'Genf', 'Trier', 'Saarbrücken', 'Lausanne'}:
            plt.text(xpt, ypt, name, fontsize="12", zorder=5)

if args.save:
    print("Saving to '%s'" % (args.save), file=sys.stderr, flush=True)
    fig.savefig(args.save, dpi=200, bbox_inches='tight', pad_inches=0.1)
    print("done", file=sys.stderr, flush=True)
else:
    fig.tight_layout()
    plt.show()
