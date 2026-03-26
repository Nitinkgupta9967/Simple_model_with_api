from fastapi import FastAPI
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from schema.client import Client
from schema.prediction_response import PredictionResponse
from Model.predict import predict_output

app = FastAPI()
model = joblib.load("Model/model.pkl")
    
@app.get("/")
def read_root():
    return {"message": "Welcome to the Insurance Premium Prediction API. Please use the /predict endpoint to get predictions."}

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "version" : "1.0.0",
        "Model Loaded" : model is not None
        }

@app.post("/predict", response_model=PredictionResponse)
def predict_insurance_premium(c: Client) -> PredictionResponse:

    bmi = c.weight/(c.height**2)
    df = pd.DataFrame([{
        "income_lpa": c.income_lpa,
        "lifestyle_risk": c.lifestyle_risk,
        "age_group": c.age_group,
        "city_tier": c.city_tier,
        "occupation": c.occupation,
        "bmi": bmi
    }])
    print(df)
    try:
        prediction = predict_output(df)
        return PredictionResponse(**{
            "insurance_premium_category": prediction["predicted_category"],
            "confidence": prediction["confidence"],
            "class_probabilities": prediction["class_probabilities"],
        })
    except Exception as e:
        return JSONResponse(status_code =500, content={"error": str(e)})

