import matplotlib
from scipy.sparse import csr_matrix

matplotlib.use('Qt5Agg')
import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gd
import pandas as pd
import seaborn
from shapely.geometry import Point

seaborn.set('paper')

cm = plt.get_cmap('gist_rainbow')
cMap = 'gist_rainbow'

country_boxes = {
    'germany': [47., 55.1, 5.5, 15.5, 50., 10., 10.],
    'german-speaking': [45.818, 55.1, 5.5, 17.2459, 50., 10., 10.],
    'austria': [46.3108, 49.0697, 9.451, 17.2459, 47.5, 13., 13.],
    'switzerland': [45.818, 47.85, 5.8887, 10.5579, 46., 7., 7.]
}

parser = argparse.ArgumentParser('put individual words on a map')
parser.add_argument('target', help='target word', nargs='+')
parser.add_argument('--corpus', help='inverted index corpus name')
parser.add_argument('--country', help='map restriction', choices={'AT', 'DE', 'CH', 'german-speaking'}, default='german-speaking')
parser.add_argument('--locations', help='location files', nargs='+')
parser.add_argument('--resolution', help='map resolution', choices={'l', 'i', 'h', 'f'}, default='l', type=str)
parser.add_argument('--restrict', help='restrict to a country, defined by ISO', choices={'AT', 'DE', 'CH'},
                    default=None, type=str)
parser.add_argument('--save', help='save as PNG file', default=None)
parser.add_argument('--shapefile', help='shapefile for map',
                    default='/Users/dirkhovy/Dropbox/working/lowlands/GeoStats/data/nuts/NUTS_RG_03M_2010.shp')
parser.add_argument('--show_cities', help='show city names on map', action='store_true')

args = parser.parse_args()

# read in locations
locations = {}
print("reading locations", file=sys.stderr, flush=True)
for location_file in args.locations:
    location_list = json.load(open(location_file))
    for location in location_list:
        try:
            if args.restrict is None or args.restrict == location['country']:
                locations[location['city']] = [(location['lat'], location['lng']), location['country']]
        except KeyError:
            # print('wrong input format! City key should be "city"')
            # sys.exit()
            if args.restrict is None or args.restrict == location['country']:
                locations[location['location']] = [(location['lat'], location['lng']), location['country']]

# retrieve
print("loading inverted index", file=sys.stderr, flush=True)
with open('%s.ii.json' % args.corpus) as json_info:
    obj = json.load(json_info)
    w2i = obj['word2int']
    c2i = obj['city2int']
i2c = {value: key for key, value in c2i.items()}
i2w = {value: key for key, value in w2i.items()}

# get word counts and totals
loader = np.load('%s.ii.npz' % args.corpus)
word_counts = csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])
# get total counts, add smoothing
totals = word_counts.sum(axis=0) + 0.000000001

targets = [w2i[w] for w in args.target if w in w2i]
missing = [w for w in args.target if w not in w2i]
for w in missing:
    print('\tWARNING:\t"%s" was not found in vocab' % w, file=sys.stderr, flush=True)

if targets == [] or len(missing) == len(targets):
    print('none of the target words found', file=sys.stderr, flush=True)
    sys.exit(0)


# sub-select rows and scale
W = word_counts[targets]
most_frequent = np.array(np.nanmax(W.todense(), axis=1)) * 50
W = W / np.array(totals)
best = np.array(np.nanmax(W, axis=1))
W = np.array(W / best) * most_frequent + 10

country = args.country

width = 8
height = 10

df = gd.read_file(args.shapefile)
if country == 'german-speaking':
    area = df[(df.STAT_LEVL_ == 0) & ((df.NUTS_ID == 'DE') | (df.NUTS_ID == 'AT') | (df.NUTS_ID == 'CH'))]
else:
    area = df[(df.STAT_LEVL_ == 0) & (df.NUTS_ID == country)]

eligible_cities = [i2c[city_id] for city_id in (totals > 10000).nonzero()[1] if i2c[city_id] in locations]
colors = [cm(1. * i / len(targets)) for i in range(len(targets))]

# plot background map
fs = (int(area.geometry.bounds.maxx.max() - area.geometry.bounds.minx.min())//1.5, int(area.geometry.bounds.maxy.max() - area.geometry.bounds.miny.min()))
fig, ax = plt.subplots(figsize=fs)
area.plot(ax=ax, edgecolor='black', facecolor='white', linewidth=1);

print(fs)
# plot each word
for w, word in enumerate(targets):
    coords = [(locations[city][0][1], locations[city][0][0]) for city in eligible_cities]

    z = W[w, [c2i[city] for city in eligible_cities]]

    gdf = pd.DataFrame()
    gdf['Coordinates'] = coords
    gdf['Coordinates'] = gdf['Coordinates'].apply(Point)
    pts = gd.GeoDataFrame(geometry=gdf.Coordinates)

    pts.plot(ax=ax, c=colors[w], zorder=4, alpha=0.6, label=i2w[word], markersize=z);

    if args.show_cities:
        top10 = [i2c[city_id] for city_id in W[w].argsort() if i2c[city_id] in locations and W[w, city_id] > 0][-10:]
        # add city names
        lons, lats = zip(*[locations[city][0] for city in top10])

        for name, xpt, ypt in zip(top10, lats, lons):
            plt.text(xpt, ypt, name, fontsize="12", zorder=5)


lgnd = plt.legend(fontsize='12', frameon=True)
for handle in lgnd.legendHandles:
    handle.set_sizes([400])

if args.save:
    print("Saving to '%s'" % (args.save), file=sys.stderr, flush=True)
    fig.savefig(args.save, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print("done", file=sys.stderr, flush=True)
else:
    fig.tight_layout()
    plt.show()
