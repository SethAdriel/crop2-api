from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model
model = joblib.load("model/crop_recommender_model.pkl")

# App
app = FastAPI(title="🌱 Crop Recommendation API")

class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.get("/")
def root():
    return {"message": "Welcome to the Crop Recommender API"}

@app.post("/predict")
def predict(data: CropInput):
    df = pd.DataFrame([[data.N, data.P, data.K, data.temperature,
                        data.humidity, data.ph, data.rainfall]],
                      columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    prediction = model.predict(df)[0]
    return {"recommended_crop": prediction}
