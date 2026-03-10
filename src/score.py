import os
import json
import joblib
import pandas as pd

def init():
    global model
    # AZUREML_MODEL_DIR is automatically created by Azure. 
    # It points to the folder where your registered model was downloaded.
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "best_model.pkl")
    model = joblib.load(model_path)
    print("Model loaded successfully!")

def run(raw_data):
    try:
        # 1. Parse the incoming JSON request
        data = json.loads(raw_data)
        
        # 2. Convert JSON into a pandas DataFrame so our pipeline can handle it
        df = pd.DataFrame(data["input_data"])
        
        # 3. Make the prediction (0 = No Churn, 1 = Churn)
        predictions = model.predict(df)
        
        # 4. Return the result back to the user
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}