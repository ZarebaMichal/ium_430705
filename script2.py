"""
Download dataset between 10-20 mb,
Split it into train/dev/test
Return dataset info (length, max, min etc.)
"""

import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import wget

url = "https://git.wmi.amu.edu.pl/s430705/ium_430705/raw/commit/755bc2b67abdfdd67dd9abac0c850d21b04d0da5/imdb_movies.csv"
#url = 'https://git.wmi.amu.edu.pl/s430705/ium_430705/raw/branch/master/imdb_movies.csv'
wget.download(url, out='imdb_movies.csv', bar=None)

movies_data = pd.read_csv("imdb_movies.csv")

# Drop rows with missing values
movies_data.dropna(inplace=True)

# Remove not interesting columns
drop_columns = ["title_id", "certificate", "title", "plot"]
movies_data.drop(labels=drop_columns, axis=1, inplace=True)

# Normalize data, lowercase str
for column_name in ["original_title", "countries", "genres", "director", "cast"]:
    movies_data[column_name] = (
        movies_data[column_name]
        .str.translate(str.maketrans("", "", string.punctuation))
        .str.lower()
    )

# Remove ',' from votes number and change type to int
movies_data["votes_number"] = (movies_data["votes_number"].str.replace(",", "")).astype(
    int
)

# Normalize number values
scaler = preprocessing.MinMaxScaler()
movies_data[["votes_number", "year", "runtime"]] = scaler.fit_transform(
    movies_data[["votes_number", "year", "runtime"]]
)

drop_columns = [
    "original_title",
    "countries",
    "genres",
    "director",
    "cast",
    "release_date",

]

movies_data.drop(labels=drop_columns, axis=1, inplace=True)

# Split set to train/dev/test 6:2:2 ratio and save to .csv file
train, dev = train_test_split(movies_data, train_size=0.6, test_size=0.4, shuffle=True)
dev, test = train_test_split(dev, train_size=0.5, test_size=0.5, shuffle=True)

train.to_csv("train.csv")
dev.to_csv("dev.csv")
test.to_csv("test.csv")

# Get length of given sets
print(f"Test dataset length: {len(test)}")
print(f"Dev dataset length: {len(dev)}")
print(f"Train dataset length: {len(train)}")
print(f"Whole dataset length: {len(movies_data)}, \n")

# Print information of given columns
for column in ["year", "rating", "runtime", "votes_number"]:
    column_data = movies_data[column]
    print(f"Information on {column}")
    print(f"Min: {column_data.min()}")
    print(f"Max: {column_data.max()}")
    print(f"Mean: {column_data.mean()}")
    print(f"Median: {column_data.median()}")
    print(f"Standard deviation: {column_data.std()}, \n")