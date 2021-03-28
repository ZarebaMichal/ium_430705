#!/bin/bash

curl -OL https://git.wmi.amu.edu.pl/s430705/ium_430705/src/branch/master/imdb_movies.csv
head -n -1 imdb_movies.csv | shuf > imdb_movies.csv.shuf
wc -l imdb_movies.csv
head -n 126 imdb_movies.csv.shuf > test.csv
head -n 126 imdb_movies.csv.shuf | tail -n 2006 > dev.csv
tail -n 379 imdb_movies.csv.shuf > train.csv
wc -l *.csv
