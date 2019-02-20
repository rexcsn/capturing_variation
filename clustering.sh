a="euclidean"
l="ward"
for c in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do 
    python -W ignore src/evaluate_Lameli_fit.py --locations data/country_stuff.json --prefix models/no_cities --model models/no_cities.model --clusters $c --min 200 --affinity $a --linkage $l
done > clustering_results.txt

for c in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do 
    python -W ignore src/evaluate_Lameli_fit.py --locations data/country_stuff.json --prefix models/no_cities --model models/no_cities.model --clusters $c --min 200 --affinity $a --linkage $l --retrofit nuts2
done > clustering_results_retrofit.txt
