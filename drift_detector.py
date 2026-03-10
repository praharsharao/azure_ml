import os
import requests
import pandas as pd
import mlflow
from scipy.stats import wasserstein_distance
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def check_drift():
    print("🔍 Starting Custom Data Drift Evaluation...")
    
    # 1. Authenticate to Azure ML
    try:
        ml_client = MLClient.from_config(credential=DefaultAzureCredential())
        tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Insurance_Data_Drift_Monitor")
    except Exception as e:
        print(f"⚠️ Could not connect to Azure MLflow: {e}")

    # 2. Load the baseline training data
    try:
        baseline_df = pd.read_csv("insurance.csv")
    except FileNotFoundError:
        print("❌ Could not find insurance.csv baseline.")
        return

    # 3. Simulate HEALTHY live data
    # We only add 1 year to the age, which is a normal shift
    live_df = baseline_df.sample(100).copy()
    live_df['age'] = live_df['age'] + 1 
    
    # 4. Calculate the Drift Score
    drift_score = wasserstein_distance(baseline_df['age'], live_df['age'])
    print(f"📊 Calculated Drift Score for 'age': {drift_score:.2f}")
    threshold = 5.0 

    # 5. LOG TO MLFLOW
    with mlflow.start_run():
        mlflow.log_metric("age_wasserstein_distance", drift_score)
        mlflow.log_param("drift_threshold", threshold)
        
        if drift_score > threshold:
            mlflow.log_param("status", "DRIFT_DETECTED_RETRAINING")
            print(f"🚨 ALERT: Data Drift ({drift_score:.2f}) exceeds threshold ({threshold})!")
            
            github_token = os.environ.get("MY_GITHUB_PAT")
            url = "https://api.github.com/repos/praharsharao/azure_ml/dispatches"
            headers = {"Authorization": f"token {github_token}", "Accept": "application/vnd.github.v3+json"}
            payload = {"event_type": "drift_alert_retrain"}
            
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 204:
                print("✅ Retraining pipeline triggered!")
        else:
            # THIS IS THE PATH THAT WILL RUN NOW
            mlflow.log_param("status", "HEALTHY")
            print(f"✅ Data is healthy (Drift: {drift_score:.2f}). No retraining required.")

if __name__ == "__main__":
    check_drift()
