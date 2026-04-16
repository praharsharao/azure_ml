import os
import pandas as pd
import mlflow
import traceback

def init():
    global model
    model_dir = os.environ["AZUREML_MODEL_DIR"]
    model_subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    model_path = os.path.join(model_dir, model_subdirs[0]) if model_subdirs else model_dir
    model = mlflow.pyfunc.load_model(model_path)

def run(mini_batch):
    batch_results = []
    
    for file_path in mini_batch:
        try:
            # 1. Load data
            data = pd.read_csv(file_path)
            
            # 2. Extract IDs safely
            if 'customer_id' in data.columns:
                ids = data['customer_id'].astype(str).values
                features = data.drop(columns=['customer_id'])
            else:
                ids = data.index.astype(str).values
                features = data
                
            # 3. Predict
            predictions = model.predict(features)
            
            # 4. Standardize output
            if isinstance(predictions, pd.DataFrame):
                preds = predictions.iloc[:, 0].astype(str).values
            else:
                preds = predictions
                
            # 5. Build success dataframe
            out_df = pd.DataFrame({
                "customer_id": ids,
                "prediction": preds
            })
            batch_results.append(out_df)
            
        except Exception as e:
            
            print(f"Skipping invalid/hidden file. Reason: {e}")
            continue 
            
    return pd.concat(batch_results)