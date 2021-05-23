import sys

import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from urllib.parse import urlparse


mlflow.set_tracking_uri("http://172.17.0.1:5000")


def prepare_model(train_size_param, test_size_param, epochs, batch_size):
    movies_data = pd.read_csv("train.csv", error_bad_lines=False)
    movies_data.drop(movies_data.columns[0], axis=1, inplace=True)
    movies_data.dropna(inplace=True)
    X = movies_data.drop("rating", axis=1)
    Y = movies_data["rating"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size_param, random_state=42
    )

    test_df = pd.read_csv("test.csv")
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

    input_example = X_test.values[10]

    rmse = mean_squared_error(y_test, y_pred)

    model.save("model_movies")

    return model, rmse, X_train, input_example


train_size_param = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
test_size_param = float(sys.argv[2]) if len(sys.argv) > 1 else 0.2
epochs = int(sys.argv[3]) if len(sys.argv) > 1 else 400
batch_size = int(sys.argv[4]) if len(sys.argv) > 1 else 128


with mlflow.start_run():

    mlflow.log_param("train size", train_size_param)
    mlflow.log_param("test size", test_size_param)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch size", batch_size)

    model, rmse, X_train, input_example = prepare_model(
        train_size_param=train_size_param,
        test_size_param=test_size_param,
        epochs=epochs,
        batch_size=batch_size,
    )

    mlflow.log_metric("RMSE", rmse)

    signature = mlflow.models.signature.infer_signature(X_train.values, model.predict(X_train.values))
    # mlflow.keras.save_model(model, "movies_imdb", input_example=input_example, signature=signature)
    mlflow.set_experiment("s430705")
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    if tracking_url_type_store != "file":
        mlflow.keras.log_model(model, "movies_imdb2", registered_model_name="s430705", signature=signature,
                                 input_example=input_example)
    else:
        mlflow.keras.log_model(model, "model_movies", signature=signature, input_example=input_example)
        mlflow.keras.save_model(model, "movies_mdb", signature=signature, input_example=input_example)
