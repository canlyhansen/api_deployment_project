from fastapi import FastAPI, HTTPException
from model.predict import get_prediction
from typing import List
import pandas as pd

app = FastAPI(title="Fraud Predictor API")

@app.get('/')
def root():
    return {"message": "API is running"}

@app.post('/predict')
def predict(request: List[dict]):
    try:
        result = get_prediction(request)
        return {'prediction': result.tolist()}
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))
    
@app.post('/predict_proba')
def predict_proba(request: List[dict]):
    try:
        result = get_prediction(request, proba=True)
        return {'prediction': result.tolist()}
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))
    
# @app.get('/schema')
# def get_schema():
#     df = pd.read_csv('data/test_2025.csv')
#     return df.dtypes.to_dict()