import os
import glob
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# 1. Initialize the FastAPI App
app = FastAPI(
    title="Insurance Churn Risk API (Auto-Syncing)",
    description="Real-time API connected natively to Azure Model Registry.",
    version="2.0.0"
)

# 2. Securely Connect to Azure & Download Latest Model
try:
    print("🔒 Authenticating with Azure...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id="fad1a96b-a67e-452d-87d8-fcdb0a781616",
        resource_group_name="sample_uc",
        workspace_name="sample_test_uc1"
    )
    
    print("☁️ Finding LATEST model version from Cloud Registry...")
    # This specifically asks Azure for the highest version number
    latest_model = ml_client.models.get(name="insurance-churn-prediction-model", label="latest")
    print(f"📥 Found Version {latest_model.version}! Downloading automatically...")
    
    # Auto-download the latest cloud model to a local cache folder
    download_dir = "./auto_downloaded_model"
    ml_client.models.download(
        name="insurance-churn-prediction-model", 
        version=latest_model.version, 
        download_path=download_dir
    )
    
    # Dynamically find the .pkl file inside the downloaded folder
    pkl_files = glob.glob(f"{download_dir}/**/*.pkl", recursive=True)
    if pkl_files:
        model = joblib.load(pkl_files[0])
        print(f"✅ Cloud Model (Version {latest_model.version}) loaded successfully into memory!")
    else:
        raise Exception("Downloaded model, but no .pkl file found inside.")
    
except Exception as e:
    print(f"⚠️ Warning: Could not load cloud model. Error: {e}")
    model = None

# 3. Define the Pydantic Schema
class BatchPayload(BaseModel):
    input_data: List[Dict[str, Any]]
    data: List[Dict[str, Any]] = None  

# 4. Create the GET Endpoint (Health Check)
@app.get("/")
def health_check():
    """Verify API is alive and show model connection status."""
    return {
        "status": "healthy", 
        "model_loaded_from_cloud": model is not None,
        "message": "Insurance API is fully synced with Azure Registry."
    }

# 5. Create the POST Endpoint (Prediction Engine)
@app.post("/predict")
def predict(payload: BatchPayload):
    if model is None:
        raise HTTPException(status_code=500, detail="Cloud Model is not loaded.")
    
    try:
        df = pd.DataFrame(payload.input_data)
        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
