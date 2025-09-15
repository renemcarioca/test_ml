FROM python:3.9-slim

WORKDIR /app

COPY model.joblib ./models/
COPY settings.json ./models/
COPY model-settings.json ./models/

RUN pip install mlserver mlserver-sklearn xgboost-cpu

EXPOSE 8080 8081

CMD ["mlserver", "start", "./models/"]