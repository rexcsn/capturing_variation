import pandas as pd
from collections import Counter, defaultdict
import numpy as np
from itertools import islice, count
from scipy.sparse import lil_matrix, csr_matrix
import sys
import json
import argparse

parser = argparse.ArgumentParser(description="compare regions")
parser.add_argument('corpus', help='input corpus file', type=str)
parser.add_argument('--output', help='output file names', type=str, required=True)
args = parser.parse_args()

word2int_count = count()
word2int = defaultdict(word2int_count.__next__)

# map city names also to ints
city2int_count = count()
city2int = defaultdict(city2int_count.__next__)

rows = []
cols = []
vals = []

print("\ncollecting counts", file=sys.stderr, flush=True)
# go through data to collect city names and words
with open(args.corpus, encoding='utf-8') as corpus:
    for line_no, line in enumerate(islice(corpus, None)):
        if line_no > 0:
            if line_no % 5000 == 0:
                print("%s" % (line_no), file=sys.stderr, flush=True)
            elif line_no % 100 == 0:
                print('.', file=sys.stderr, end=' ', flush=True)

        elements = json.loads(line.strip())
        city = city2int[elements['tags'][0]]
        words = [word2int[word] for word in elements['words']]

        for wid, freq in Counter(words).items():
            rows.append(wid)
            cols.append(city)
            vals.append(freq)

# create sparse word-by-count matrix
print("\ncreating sparse count matrix", file=sys.stderr, flush=True)
counts = csr_matrix((vals, (rows, cols)), dtype=np.int, shape=(len(word2int), len(city2int)))

values = counts.data
rows = counts.indices
cols = counts.indptr

print("\nsaving counts", file=sys.stderr, flush=True)
np.savez('%s.ii' % args.output, data=values, indices=rows, indptr=cols, shape=(len(word2int), len(city2int)))

print("\nsaving auxiliary info", file=sys.stderr, flush=True)
with open('%s.ii.json' % args.output, 'w') as dict_file:
    obj = {
        'word2int': dict(word2int),
        'city2int': dict(city2int),
    }
    json.dump(obj, dict_file, ensure_ascii=False)

print("\nall done!", file=sys.stderr, flush=True)
