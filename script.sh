#!/bin/bash

curl -LO https://git.wmi.amu.edu.pl/s430705/ium_430705/raw/branch/master/imdb_movies.csv
head -n -1 imdb_movies.csv | tail -n +$((${CUTOFF}+1)) > imdb_movies.csv | shuf > imdb_movies.csv.shuf
wc -l imdb_movies.csv
head -n 4600 imdb_movies.csv.shuf > test.csv
head -n 4600 imdb_movies.csv.shuf | tail -n 4600 > dev.csv
tail -n 13800 imdb_movies.csv.shuf > train.csv
wc -l *.csv
