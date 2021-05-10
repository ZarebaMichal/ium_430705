from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from tensorflow.keras.models import load_model
import pandas as pd

test_df = pd.read_csv('test.csv')
test_df.drop(test_df.columns[0], axis=1, inplace=True)
x_test = test_df.drop("rating", axis=1)
y_test = test_df["rating"]

model = Sequential()
model = load_model('model_movies')

y_pred = model.predict(x_test.values)

rmse = mean_squared_error(y_test, y_pred)

print(f"RMSE: {rmse}")
