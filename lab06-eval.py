import sys

import pandas as pd
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

test_df = pd.read_csv("test.csv")
test_df.drop(test_df.columns[0], axis=1, inplace=True)
x_test = test_df.drop("rating", axis=1)
y_test = test_df["rating"]

model = load_model("model_movies")
y_pred = model.predict(x_test.values)

rmse = mean_squared_error(y_test, y_pred)
build_number = sys.argv[1] if len(sys.argv) > 1 else 0

d = {"rmse": [rmse], "build": [build_number]}
df = pd.DataFrame(data=d)

with open("evaluation.csv", "a") as f:
    df.to_csv(f, header=f.tell() == 0, index=False)
