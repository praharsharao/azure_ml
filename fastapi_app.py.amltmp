import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# 1. Initialize the FastAPI App
app = FastAPI(
    title="Insurance Churn Risk API",
    description="Real-time and batch scoring API for customer churn prediction.",
    version="1.0.0"
)

# 2. Load the Model at Startup (Global Scope)
# Ensure 'best_model.pkl' is in the same directory as this script!
MODEL_PATH = "outputs/best_insurance_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load model. Ensure {MODEL_PATH} exists. Error: {e}")
    model = None

# 3. Define the Pydantic Schema
# This strictly enforces that the API receives the exact JSON structure Streamlit sends
class BatchPayload(BaseModel):
    input_data: List[Dict[str, Any]]
    data: List[Dict[str, Any]] = None  # Optional fallback key

# 4. Create the GET Endpoint (Health Check)
@app.get("/")
def health_check():
    """Manager Requirement: A GET endpoint to verify the API is alive."""
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "message": "Insurance API is running."
    }

# 5. Create the POST Endpoint (Prediction Engine)
@app.post("/predict")
def predict(payload: BatchPayload):
    """Manager Requirement: A POST endpoint to accept data and return predictions."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
    
    try:
        # Step A: Convert the validated JSON payload into a Pandas DataFrame
        df = pd.DataFrame(payload.input_data)
        
        # Step B: Generate predictions (0 = No Churn, 1 = Churn)
        predictions = model.predict(df)
        
        # Step C: Return the exact dictionary format Streamlit expects
        return {"predictions": predictions.tolist()}
    
    except Exception as e:
        # If anything goes wrong, return a clean 400 Bad Request error
        raise HTTPException(status_code=400, detail=str(e))