import json
import mlflow
import pandas as pd
from pprint import pprint
from mlflow.tracking import MlflowClient

model_name = "s434804"
model_version = 12
 
mlflow.set_tracking_uri("http://172.17.0.1:5000")

model = mlflow.keras.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}",
)

"""
Honestly I have no clue how to get model's json input example
any other way, so just intialize MLFlow client, get latest
version of choosen model, and then get path to model's files
"""

client = MlflowClient()
models_version = client.search_model_versions("name='s434804'")
path_to_input = models_version[-1].source

with open(f'{path_to_input}/input_example.json', 'r') as datafile:
    data = json.load(datafile)
    example_input = data["inputs"]

input_dictionary = {i: x for i, x in enumerate(example_input)}
input_ex = pd.DataFrame(input_dictionary, index=[0])
print(model.predict(input_ex))
