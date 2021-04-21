import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import wget
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error


url = "https://git.wmi.amu.edu.pl/s430705/ium_430705/raw/branch/master/imdb_movies.csv"
wget.download(url, out="imdb_movies.csv", bar=None)

movies_data = pd.read_csv("imdb_movies.csv")

# Drop rows with missing values
movies_data.dropna(inplace=True)

#ToDo: Prepare columns for actors, genres, countries

# Remove not interesting columns
drop_columns = [
    "title_id",
    "certificate",
    "title",
    "plot",
    "original_title",
    "countries",
    "genres",
    "director",
    "cast",
    "release_date",
    "certificate",
    "plot",
]
movies_data.drop(labels=drop_columns, axis=1, inplace=True)

# Normalize data, lowercase str
# for column_name in ["original_title", "countries", "genres", "director", "cast"]:
#     movies_data[column_name] = (
#         movies_data[column_name]
#         .str.translate(str.maketrans("", "", string.punctuation))
#         .str.lower()
#     )

# Remove ',' from votes number and change type to int
movies_data["votes_number"] = (movies_data["votes_number"].str.replace(",", "")).astype(
    int
)

# Normalize number values
scaler = preprocessing.MinMaxScaler()
movies_data[["votes_number", "year", "runtime"]] = scaler.fit_transform(
    movies_data[["votes_number", "year", "runtime"]]
)


X = movies_data.drop("rating", axis=1)
Y = movies_data["rating"]

# Split set to train/test 8:2 ratio
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Set up model
model = Sequential()
model.add(Dense(8, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(3, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)


model.fit(
    x=X_train,
    y=Y_train.values,
    validation_data=(X_test, Y_test.values),
    batch_size=128,
    epochs=400,
    callbacks=[early_stop],
)

# Predict movie ratings
predictions = model.predict(X_test)

pd.DataFrame(predictions).to_csv('results.csv')


# Compare outputs
for i, score in enumerate(predictions):
    print(f"Original score: {Y_test.iloc[i]} Predicted score: {score} \n")
    print(f"Difference is : {Y_test.iloc[i] - score}")


# Evaluate
print(mean_absolute_error(Y_test, predictions))

