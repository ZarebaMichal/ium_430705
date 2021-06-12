import json
import mlflow
import pandas as pd

import mlflow.pyfunc

model_name = "country_vaccination"
stage = '1'

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{stage}"
)


with open('country_vaccination/input_example.json', 'r') as datafile:
    data = json.load(datafile)
    example_input = data["inputs"]

input_dictionary = {i : x for i, x in enumerate(example_input) }
input_ex = pd.DataFrame(input_dictionary, index=[0])
print(model.predict(input_ex))
