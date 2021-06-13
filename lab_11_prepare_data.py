"""
Download dataset between 10-20 mb,
Split it into train/dev/test
Return dataset info (length, max, min etc.)
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

movies_data = pd.read_csv("imdb_movies.csv")

# Drop rows with missing values
movies_data.dropna(inplace=True)

# Remove not interesting columns
drop_columns = ["title_id", "certificate", "title", "plot"]
drop_columns2 = [
    "original_title",
    "countries",
    "genres",
    "director",
    "cast",
    "release_date",
]
drop_columns = drop_columns + drop_columns2

movies_data.drop(labels=drop_columns, axis=1, inplace=True)


# Remove ',' from votes number and change type to int
movies_data["votes_number"] = (movies_data["votes_number"].str.replace(",", "")).astype(
    int
)

# Normalize number values
scaler = preprocessing.MinMaxScaler()
movies_data[["votes_number", "year", "runtime"]] = scaler.fit_transform(
    movies_data[["votes_number", "year", "runtime"]]
)

# Split set to train/dev/test 6:2:2 ratio and save to .csv file
train, dev = train_test_split(movies_data, train_size=0.6, test_size=0.4, shuffle=True)
dev, test = train_test_split(dev, train_size=0.5, test_size=0.5, shuffle=True)

train.to_csv("train.csv")
dev.to_csv("dev.csv")
test.to_csv("test.csv")
