#!/bin/bash

curl -LO https://git.wmi.amu.edu.pl/s430705/ium_430705/raw/branch/master/imdb_movies.csv
shuf ./imdb_movies.csv > ./imdb_movies2.csv | tail -n +5 > ./imdb_movies2.csv
wc -l ./imdb_movies.csv
head -n 4600 ./imdb_movies2.csv > ./test.csv
head -n 4600 ./imdb_movies2.csv | tail -n 4600 > ./dev.csv
tail -n 13800 ./imdb_movies2.csv > ./train.csv
wc -l ./*.csv
