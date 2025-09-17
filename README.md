# Teste de conhecimentos - ML Engineer
Respositório público de código para o teste de conhecimentos da vaga de Engenheiro(a) de Machine Learning.

## Problema escolhido: Santander Customer Transaction Prediction
Classificação binária com 200 atributos anonimizados de transações e 200000 entradas. O data card no Kaggle mostra que há um desbalanço nas labels e que todos os atributos são numéricos distribuídos normalmente, também verifiquei em código que não há dados faltantes. Apenas o arquivo `train.csv` foi usado para treinamento e avaliação iniciais porque o arquivo `test.csv` não contém a variável alvo. A partir do arquivo `train.csv` os conjuntos de treino e teste foram criados de um corte 80/20 aleatório estratificado de forma que ambos tenham uma mesma proporção de labels positivas e negativas próxima da original.

## Ambiente
Esse trabalho foi conduzido num ambiente Conda com especificações no arquivo `requirements.txt` e pode ser reproduzido usando o comando 

```
$ conda create --name <env> --file requirements.txt
```

## Treinamento de modelo
O código em `ml_code.ipynb` inclui o carregamento dos dados e seleção de modelo e hiperparâmetros de uma coleção pequena de modelos paramétricos e não paramétricos de diferentes complexidades com validação cruzada por 5 folds estratificados. Por causa do forte desbalanço entre labels positivas e negativas, a métrica utilizada para seleção foi F1 Score porque seu cálculo penaliza performance ruim em classes menos frequentes onde acurácia pode não fazê-lo. O conjunto modelo/hiperparâmetros finalmente selecionado é treinado no conjunto de treino inteiro e avaliado em uma seleção ampla de métricas no conjunto de teste e serializado para deploy no arquivo `model.joblib`.

## Deploy
A metodologia selecionada para deploy foi o uso de conteinerização, com a criação de container Docker onde o modelo é servido com MLServer, com todos os arquivos relevantes estando na pasta `deploy`. Os arquivos `settings.json` e `model-settings.json` contêm as principais informações sobre a configuração do servidor, onde `max_batch_size` e `max_batch_time` em `model-settings.json` determinam respectivamente os máximos de requisições em fila e tempo passado em segundos para processamento de requisições em batch, e podem ser otimidos para suportar apenas predições instantâneas. `Dockerfile` detalha a criação de uma imagem Docker que pode ser instanciada para a criação do modelo.

O deploy do modelo é realizado pelos seguintes comandos que disponibilizarão o modelo para requisições no url `http://localhost:8080/v2/models/xgboost-model/infer`:
```
$ docker build -t mlserver-model-deploy
$ docker run -p 8080:8080 mlserver-model-deploy
```

E a execução do script `request.py` faz uma requisição simples ao modelo.
