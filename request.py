import requests
import json
import pandas as pd
from mlserver.codecs import PandasCodec

url = "http://localhost:8080/v2/models/xgboost-model/infer"
size = 100

test_data = pd.read_csv('santander-customer-transaction-prediction/test.csv').iloc[:, 1:]
test_data = test_data.sample(n=size)

inference_request = PandasCodec.encode_request(test_data)

payload = inference_request.dict()

try:
    response = requests.post(url=url, json=payload)

    result = response.json()

    print("Response from MLServer:")
    print(json.dumps(result, indent=4))

    predictions = result.get("outputs", [])
    if predictions:
        prediction_result = predictions[0]["data"][0]
        print(f"\nPrediction: {prediction_result}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
