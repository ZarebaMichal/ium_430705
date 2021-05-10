import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from sacred.observers import FileStorageObserver
from sacred import Experiment

ex = Experiment("file_observer", interactive=False, save_git_info=False)
ex.observers.append(FileStorageObserver('lab07/my_runs'))


@ex.config
def my_config():
    train_size_param = 0.8
    test_size_param = 0.2
    epochs = 400
    batch_size = 128


@ex.capture
def prepare_model(train_size_param, test_size_param, epochs, batch_size, _run):
    _run.info["prepare_model_ts"] = str(datetime.now())
    movies_data = pd.read_csv('train.csv', error_bad_lines=False)
    movies_data.drop(movies_data.columns[0], axis=1, inplace=True)
    movies_data.dropna(inplace=True)
    X = movies_data.drop("rating", axis=1)
    Y = movies_data["rating"]

    print(X, Y.values)
    # Split set to train/test 8:2 ratio
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size_param, random_state=42
    )

    test_df = pd.read_csv('test.csv')
    test_df.drop(test_df.columns[0], axis=1, inplace=True)
    x_test = test_df.drop("rating", axis=1)
    y_test = test_df["rating"]

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
        x=X_train.values,
        y=Y_train.values,
        validation_data=(X_test, Y_test.values),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stop],
    )
    y_pred = model.predict(x_test.values)

    rmse = mean_squared_error(y_test, y_pred)

    _run.info["Final Results: "] = rmse

    model.save('model_movies')

    return rmse


@ex.automain
def my_main(train_size_param, test_size_param, epochs, batch_size):
    print(prepare_model())


r = ex.run()
ex.add_artifact("model_movies/saved_model.pb")
