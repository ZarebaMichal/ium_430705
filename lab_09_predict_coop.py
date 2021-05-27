import json
import mlflow
import pandas as pd
from pprint import pprint
from mlflow.tracking import MlflowClient

model_name = "s430705"
model_version = 30
 
mlflow.set_tracking_uri("http://172.17.0.1:5000")

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)
client = MlflowClient()
for mv in client.search_model_versions("name='s430705'"):
    pprint(dict(mv), indent=4)

with open('/tmp/mlruns/0/6be4f90846214df8913a553bc53b1019/artifacts/movies_imdb2/input_example.json', 'r') as datafile:
    data = json.load(datafile)
    example_input = data["inputs"]

input_dictionary = {i: x for i, x in enumerate(example_input)}
input_ex = pd.DataFrame(input_dictionary, index=[0])
print(model.predict(input_ex))
