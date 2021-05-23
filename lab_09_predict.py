import json
import mlflow
import pandas as pd


model = mlflow.keras.load_model("country_vaccination")


with open('country_vaccination/input_example.json', 'r') as datafile:
    data = json.load(datafile)
    example_input = data["inputs"]

input_dictionary = {i : x for i, x in enumerate(example_input) }
input_ex = pd.DataFrame(input_dictionary, index=[0])
print(model.predict(input_ex))
