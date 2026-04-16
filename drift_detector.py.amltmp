import os
import requests
import pandas as pd
import mlflow
from scipy.stats import wasserstein_distance
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def check_drift():
    print(" Starting Custom Data Drift Evaluation...")
    
    try:
        ml_client = MLClient.from_config(credential=DefaultAzureCredential())
        tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Insurance_Data_Drift_Monitor")
    except Exception as e:
        print(f" Could not connect to Azure MLflow: {e}")

    # Load Baseline
    try:
        baseline_df = pd.read_csv("insurance.csv")
    except FileNotFoundError:
        print(" Could not find insurance.csv baseline.")
        return

    # Load the Live Dummy Data
    print(" Ingesting real-time live data from Streamlit API...")
    try:
        live_df = pd.read_csv("dummy_drift_data.csv")
    except FileNotFoundError:
        print(" Could not find dummy_drift_data.csv.")
        return
    
    # Calculate Drift Score
    drift_score = wasserstein_distance(baseline_df['age'], live_df['age'])
    print(f" Calculated Drift Score for 'age': {drift_score:.2f}")
    threshold = 5.0 

    with mlflow.start_run():
        mlflow.log_metric("age_wasserstein_distance", drift_score)
        mlflow.log_param("drift_threshold", threshold)
        
        if drift_score > threshold:
            mlflow.log_param("status", "DRIFT_DETECTED_RETRAINING")
            print(" ALERT: Data Drift Detected!")
            
            github_token = os.environ.get("MY_GITHUB_PAT")
            if not github_token:
                print(" ERROR: GitHub token not found.")
                return

            # Note: Ensure this URL matches your actual GitHub username
            url = "https://api.github.com/repos/praharsharao/azure_ml/dispatches"
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "Authorization": f"token {github_token}" 
            }
            payload = {"event_type": "drift_alert_retrain"}
            
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 204:
                print(" Retraining pipeline triggered successfully via GitHub Actions!")
            else:
                print(f" Failed to trigger GitHub Actions. Status: {response.status_code}")
        else:
            mlflow.log_param("status", "HEALTHY")
            print(" Data is healthy. No drift detected.")

if __name__ == "__main__":
    check_drift()
