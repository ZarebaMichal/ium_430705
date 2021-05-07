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


movies_data = pd.read_csv('train.csv' error_bad_lines=False)
movies_data.drop(movies_data.columns[0], axis=1, inplace=True)
movies_data.dropna(inplace=True)
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


model.save('model_movies')