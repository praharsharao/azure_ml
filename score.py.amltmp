
import json
import joblib
import numpy as np
import pandas as pd
import os
import logging

def init():
    global model
    model_root = os.getenv("AZUREML_MODEL_DIR")
    model_path = None
    
    # 1. Search for the .pkl file recursively
    for root, dirs, files in os.walk(model_root):
        for file in files:
            if file.endswith("best_insurance_model.pkl"):
                model_path = os.path.join(root, file)
                break
        if model_path:
            break
            
    if model_path:
        logging.info(f"✅ Found model at: {model_path}")
        try:
            model = joblib.load(model_path)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"❌ Failed to load model: {e}")
            raise
    else:
        logging.error(f"❌ Could not find 'best_insurance_model.pkl' in {model_root}")
        # List files to help debugging in logs
        for root, dirs, files in os.walk(model_root):
            logging.info(f"Files in {root}: {files}")
        raise Exception("Model file not found!")

def run(raw_data):
    try:
        data = json.loads(raw_data)
        df = pd.DataFrame(data['data'])
        predictions = model.predict(df)
        return predictions.tolist()
    except Exception as e:
        return {"error": str(e)}