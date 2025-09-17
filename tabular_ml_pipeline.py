import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.model_selection._search import BaseSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from typing import Tuple, Union, Type, List
from os import path

def load_data(data_dir: str, data_file: str, id_col: str=None) -> pd.DataFrame:
    """
    Carregamento de dados de um arquivo csv `data_file` em um diretório `data_dir`.
    """
    
    file_path = path.join(data_dir, data_file)

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Erro: Arquivo {data_file} não encontrado em {data_dir}.")
    
    print("Dados carregados com sucesso.")
    
    if id_col is not None:
        assert id_col in df, f"Coluna de ID {id_col} informada não existe no DataFrame."
        df = df.drop(columns=id_col)
    
    return df

def split_data(df: pd.DataFrame, train_size: float, target: str, stratify_col: str=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separa dados de um DataFrame Pandas `df` em treino/teste na proporção `train_size`.
    
    Opcionalmente, usa a coluna `stratify_col` do DataFrame para separação estratificada, conservando a distribuição daquela coluna.
    """
    
    assert not df.empty, "DataFrame de entrada vazio."
    assert target in df, f"Coluna {target} não encontrada em DataFrame."
    assert train_size > 0 and train_size < 1, f"Valor de train_size {train_size} não está em (0,1)."

    if stratify_col is not None:
        assert stratify_col in df, f"Coluna {stratify_col} informada para estratificação não existe no DataFrame."
        stratify = df[stratify_col]

    X, y = df.drop(columns=target), df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=stratify, random_state=42)
    print("Dados separados em treino/teste.")
    return X_train, X_test, y_train, y_test

def create_pipeline(imputer: Union[SimpleImputer, KNNImputer], 
                    scaler: Union[StandardScaler, MinMaxScaler], 
                    model: BaseEstimator) -> Pipeline:
    """
    Cria pipeline SKLearn com objetos de imputação `imputer`, normalização `scaler` e modelo `model`.
    """
    return Pipeline([
        ('imputer', imputer),
        ('scaler', scaler),
        ('model', model)
    ])

def search_single_model(pipeline: Pipeline, 
                       param_grid: dict, 
                       cv: BaseCrossValidator, 
                       searcher: Type[BaseSearchCV],
                       scoring: str,
                       X: pd.DataFrame,
                       y: pd.Series) -> BaseSearchCV:
    """
    Raliza busca de hiperparâmetro com validação cruzada em pipeline de modelo único.
    """
    search = searcher(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        verbose=1
    )
    print("Busca montada.")
    search.fit(X, y)
    print("Busca finalizada.")
    print(f"Melhores hiperparâmetros: {search.best_params_}.")
    print(f"Melhor score: {search.best_score_}.")
    return search

def search_models(model_configs: List[dict],
                  cv: BaseCrossValidator,
                  searcher: Type[BaseSearchCV],
                  scoring: str,
                  X: pd.DataFrame,
                  y: pd.Series) -> Pipeline:
    """
    Realiza busca de hiperparâmetro em todos os modelos e seleciona o pipeline com conjunto modelo/hiperparâmetro de melhor performance.
    """
    assert not X.empty, "DataFrame de atributos vazio."
    assert not y.empty, "Series de variável alvo vazio."

    best_estimator = None
    best_score = -1
    best_params = {}
    best_model_name = ""
    print(f"Realizando busca completa para modelos: {list(map(lambda x: x['name'], model_configs))}.")
    for config in model_configs:
            model_name = config['name']
            imputer = config['imputer']
            scaler = config['scaler']
            model = config['model']
            param_grid = config['param_grid']
            pipeline = create_pipeline(
                imputer=imputer,
                scaler=scaler,
                model=model
            )
            print(f"Pipeline para {model_name} montado")
            search = search_single_model(
                pipeline=pipeline,
                param_grid=param_grid,
                cv=cv,
                searcher=searcher,
                scoring=scoring,
                X=X,
                y=y
            )
            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_params = search.best_params_
                best_estimator = search.best_estimator_
                best_model_name = model_name
    print(f"Melhor modelo encontrado {best_model_name}.")
    print(f"Melhores hiperparâmetros: {best_params}.")
    print(f"Melhor score: {best_score}.")
    return best_estimator

def final_evaluation(estimator: Pipeline, X: pd.DataFrame, y: pd.Series) -> None:
    """
    Avalia performance em maior variedade de métricas.
    """
    pred = estimator.predict(X)
    pred_proba = estimator.predict_proba(X)
    pred_proba = pred_proba[:, 1]
    print("Metrics")
    print(f"Accuracy: {accuracy_score(y, pred)}.")
    print(f"F1: {f1_score(y, pred)}.")
    print(f"Precision: {precision_score(y, pred)}.")
    print(f"Recall: {recall_score(y, pred)}.")
    print("ROC Curve")
    fpr, tpr, thresholds = roc_curve(y, pred_proba)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    plt.show()
    print("Confusion Matrix")
    c_matrix = confusion_matrix(y, pred)
    display = ConfusionMatrixDisplay(c_matrix)
    display.plot()
    plt.show()