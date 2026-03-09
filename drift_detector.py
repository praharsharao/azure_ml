import os
import requests
import pandas as pd
from scipy.stats import wasserstein_distance

def check_drift():
    print("🔍 Starting Custom Data Drift Evaluation...")
    
    try:
        baseline_df = pd.read_csv("insurance.csv")
    except FileNotFoundError:
        print("❌ Could not find insurance.csv baseline.")
        return

    live_df = baseline_df.sample(100).copy()
    live_df['age'] = live_df['age'] + 20
    
    drift_score = wasserstein_distance(baseline_df['age'], live_df['age'])
    print(f"📊 Calculated Drift Score for 'age': {drift_score:.2f}")
    
    threshold = 5.0 
    if drift_score > threshold:
        print("🚨 ALERT: Data Drift Detected!")
        
        # This securely pulls the password from GitHub Secrets instead of the code!
        github_token = os.environ.get("MY_GITHUB_PAT")
        if not github_token:
            print("❌ ERROR: GitHub token not found.")
            return

        url = "https://api.github.com/repos/praharsharao/azure_ml/dispatches"
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {github_token}" 
        }
        payload = {"event_type": "drift_alert_retrain"}
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 204:
            print("✅ Retraining pipeline triggered!")
        else:
            print(f"❌ Failed to trigger. Status: {response.status_code}")

if __name__ == "__main__":
    check_drift()
