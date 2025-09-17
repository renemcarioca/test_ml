import pandas as pd
from sklearn.pipeline import Pipeline
import os
import io
import sys

# Adiciona o diretório pai ao sys.path para encontrar os módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tabular_ml_pipeline import load_data, split_data, search_single_model

def test_load_data():
    print("\nIniciando testes para `load_data`...")
    
    try:
        csv_data = "ID_code,target,feature_1\n\
                    1,0,10\n\
                    2,1,20"
        with open("dummy_data.csv", "w") as f:
            f.write(csv_data)
        
        df = load_data(data_dir=os.getcwd(), data_file="dummy_data.csv", id_col="ID_code")
        
        assert "ID_code" not in df.columns, "Falha: Coluna de ID não foi removida."
        assert df.shape == (2, 2), "Falha: Dimensões do DataFrame incorretas."
        print("Teste de `load_data` com sucesso: OK")
    except AssertionError as e:
        print(f"Falha no teste de `load_data`: {e}")
    finally:
        if os.path.exists("dummy_data.csv"):
            os.remove("dummy_data.csv")
            
def test_split_data():
    print("\nIniciando testes para `split_data`...")
    
    try:
        data = pd.DataFrame({'feature': range(100), 'target': ([0] * 80) + ([1] * 20)})
        X_train, X_test, y_train, y_test = split_data(data, train_size=0.8, target='target', stratify_col='target')
        
        assert len(X_train) == 80, "Falha: Tamanho do conjunto de treino incorreto."
        assert len(X_test) == 20, "Falha: Tamanho do conjunto de teste incorreto."
        
        assert abs(y_train.mean() - 0.2) < 0.05, "Falha: Estratificação do treino incorreta."
        assert abs(y_test.mean() - 0.2) < 0.05, "Falha: Estratificação do teste incorreta."

        print("Teste de `split_data` com sucesso: OK")
    except AssertionError as e:
        print(f"Falha no teste de `split_data`: {e}")

def test_search_single_model():
    print("\nIniciando testes para `search_single_model`...")
    
    try:
        class MockGridSearchCV:
            def __init__(self, estimator, param_grid, cv, scoring, verbose):
                self.best_score_ = 0.95
                self.best_params_ = {'param_mock': 'value_mock'}
                self.best_estimator_ = estimator
            def fit(self, X, y):
                pass
        
        pipeline = Pipeline([('imputer', object())])
        X = pd.DataFrame({'feature_1': [1,2], 'feature_2': [3,4]})
        y = pd.Series([0,1])
        
        search = search_single_model(
            pipeline=pipeline, param_grid={}, cv=None, searcher=MockGridSearchCV, scoring='accuracy', X=X, y=y
        )
        
        assert search.best_score_ == 0.95, "Falha: Score retornado incorreto."
        assert search.best_params_ == {'param_mock': 'value_mock'}, "Falha: Parâmetros retornados incorretos."

        print("Teste de `search_single_model` com sucesso: OK")
    except AssertionError as e:
        print(f"Falha no teste de `search_single_model`: {e}")

def run_all_ml_tests():
    print("--- Executando suíte de testes do Pipeline de ML ---")
    test_load_data()
    test_split_data()
    test_search_single_model()
    print("--- Suíte de testes do Pipeline de ML concluída ---")

if __name__ == '__main__':
    run_all_ml_tests()
