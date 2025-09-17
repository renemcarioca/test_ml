import pandas as pd
import requests
from requests.exceptions import RequestException
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from request import send_inference_request

def test_send_request_success():
    print("\nIniciando teste de requisição com sucesso...")
    
    original_requests_post = requests.post
    
    try:
        class MockResponse:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self.json_data = json_data
            def json(self): return self.json_data
            def raise_for_status(self):
                if self.status_code >= 400: raise RequestException("Erro HTTP simulado")

        def mock_post_success(url, json):
            success_payload = {"outputs": [{"data": [0.1, 0.9]}]}
            return MockResponse(200, success_payload)

        requests.post = mock_post_success

        test_data = pd.DataFrame({'var_0': [1.2, 3.4], 'var_1': [5.6, 7.8]})
        response = send_inference_request("fake_url", test_data)
        
        assert response is not None, "Falha: Resposta da requisição deveria não ser nula."
        assert response['outputs'][0]['data'] == [0.1, 0.9], "Falha: Predição incorreta."
        print("Teste de requisição com sucesso: OK")
    except AssertionError as e:
        print(f"Falha no teste de requisição com sucesso: {e}")
    finally:
        requests.post = original_requests_post

def test_send_request_failure():
    print("\nIniciando teste de requisição com falha...")
    
    original_requests_post = requests.post
    
    try:
        def mock_post_failure(url, json):
            raise RequestException("Erro de conexão simulado")
        
        requests.post = mock_post_failure
        
        test_data = pd.DataFrame({'var_0': [1.2, 3.4], 'var_1': [5.6, 7.8]})
        response = send_inference_request("fake_url", test_data)
        
        assert response is None, "Falha: A resposta deveria ser nula em caso de falha."
        print("Teste de requisição com falha: OK")
    except AssertionError as e:
        print(f"Falha no teste de requisição com falha: {e}")
    finally:
        requests.post = original_requests_post

def run_all_request_tests():
    print("--- Executando suíte de testes de Requisição ---")
    test_send_request_success()
    test_send_request_failure()
    print("--- Suíte de testes de Requisição concluída ---")

if __name__ == '__main__':
    run_all_request_tests()
