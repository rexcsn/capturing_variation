mkdir -p maps
for c in 2 3 4 5 6 7 8 9 10 11 12 13
do 
    python src/map.py --locations data/country_stuff.json --corpus models/no_cities.citycounts --model models/no_cities.model --clusters $c --min 200 --show_cities --resolution h --retrofit nuts2 --restrict all --save maps/no_cities.min200.${c}clusters.GSA.RETROFIT-NUTS2.png
done
