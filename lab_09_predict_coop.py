import json
import mlflow
import pandas as pd


model_name = "s430705"
model_version = 8

mlflow.set_tracking_uri("http://172.17.0.1:5000")

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

with open('movies_imdb2/input_example.json', 'r') as datafile:
    data = json.load(datafile)
    example_input = data["inputs"]

input_dictionary = {i: x for i, x in enumerate(example_input)}
input_ex = pd.DataFrame(input_dictionary, index=[0])
print(model.predict(input_ex))
