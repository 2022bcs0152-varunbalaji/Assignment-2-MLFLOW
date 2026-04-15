from fastapi import FastAPI
from src.inference import predict

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Churn Model Running"}

@app.post("/predict")
def get_prediction(data: dict):
    result = predict(data)
    return {"churn": result}