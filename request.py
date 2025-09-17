import requests
import json
import pandas as pd
from mlserver.codecs import PandasCodec
from typing import Dict, Any

def send_inference_request(url: str, data: pd.DataFrame) -> Dict[str, Any] | None:
    """
    Prepara um DataFrame Pandas para uma requisição de inferência ao MLServer e retorna a resposta.
    """
    try:
        inference_request = PandasCodec.encode_request(data)
        payload = inference_request.dict()
        
        response = requests.post(url=url, json=payload)
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Ocorreu um erro ao fazer a requisição: {e}")
        return None

if __name__ == "__main__":
    url = "http://localhost:8080/v2/models/xgboost-model/infer"
    size = 100
    
    try:
        test_data = pd.read_csv('santander-customer-transaction-prediction/test.csv').iloc[:, 1:].sample(n=size)
    except FileNotFoundError:
        raise FileNotFoundError(f"Erro: Arquivo test.csv não encontrado.")
    
    result = send_inference_request(url, test_data)
    
    if result:
        print("Resposta do MLServer:")
        print(json.dumps(result, indent=4))
        predictions = result.get("outputs", [])
        if predictions:
            prediction_result = predictions[0]["data"][0]
            print(f"\nPredição: {prediction_result}")
    else:
        print("Não foi possível obter uma resposta válida do modelo.")
