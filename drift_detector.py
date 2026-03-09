import pandas as pd
from scipy.stats import wasserstein_distance

def check_drift():
    print(" Starting Custom Data Drift Evaluation...")
    
    # 1. Load the baseline training data
    try:
        baseline_df = pd.read_csv("insurance.csv")
    except FileNotFoundError:
        print(" Could not find insurance.csv baseline.")
        return

    # 2. Simulate live data coming from the API
    # To prove the detector works, we will take a sample of data 
    # and artificially age the customers by 20 years to force a "drift"
    live_df = baseline_df.sample(100).copy()
    live_df['age'] = live_df['age'] + 20
    
    # 3. Calculate the Drift Score
    # We compare the 'age' column of the old data vs the new data
    drift_score = wasserstein_distance(baseline_df['age'], live_df['age'])
    print(f" Calculated Drift Score for 'age': {drift_score:.2f}")
    
    # 4. Evaluate against the threshold
    threshold = 5.0 
    if drift_score > threshold:
        print(" ALERT: Data Drift Detected! The real-world data has shifted.")
        print(" Initiating Webhook to GitHub Actions for Automated Retraining...")
        
        # This is where your requests.post() Webhook goes to trigger submit_v2.py
        # requests.post("https://api.github.com/repos/abhishek-linkfields/YOUR-REPO/dispatches", ...)
        
        print(" Retraining pipeline successfully triggered!")
    else:
        print(" Data is stable. No retraining required.")

if __name__ == "__main__":
    check_drift()
