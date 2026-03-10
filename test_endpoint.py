import urllib.request
import json
import os
import ssl
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def allowSelfSignedHttps(allowed):
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

def main():
    print("🔐 Connecting to Azure...")
    ml_client = MLClient.from_config(DefaultAzureCredential())

    # We know the exact name, so we just target it directly
    endpoint_name = "insurance-churn-live-api"
    print(f"🎯 Targeting Endpoint: {endpoint_name}")

    # 1. Get Secrets Directly
    try:
        keys = ml_client.online_endpoints.get_keys(name=endpoint_name)
        auth_key = keys.primary_key
        
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        scoring_uri = endpoint.scoring_uri
    except Exception as e:
        print(f"❌ Failed to get keys/URI. Error: {e}")
        return

    print(f"🌍 Scoring URL: {scoring_uri}")
    print(f"🔑 Auth Key: {auth_key[:5]}... (hidden)")

    # 2. Prepare Data
    data = {
      "input_data": [
        {
          "age": 71, "gender": "Male", "income_band": "Upper-Middle", "employment_status": "Self-Employed",
          "province": "North West", "urban_rural": "Urban", "household_size": 6, "province_risk_score": 0.26,
          "policy_type": "Accident", "tenure_months": 90, "num_active_policies": 3, "sum_assured": 319973.92,
          "monthly_premium": 1798.49, "expected_annual_premium": 21581.88, "industry_type": "Manufacturing",
          "num_employees": 48, "location": "Urban", "safety_score": 71, "coverage_type": "Accident",
          "deductible": 1075.61, "policy_age_months": 90, "job_role": "Driver", "employee_age": 71,
          "health_score": 47, "risk_score": 59.0, "risk_level": "Medium", "past_claims_count": 2,
          "past_claims_amount": 88912.14, "expected_loss": 74737.01, "premium_recommendation": 85217.98,
          "payment_method": "Debit Order", "debit_order_flag": 1, "missed_payments_12m": 0, "lapse_flag": 0,
          "claims_count_12m": 0, "claim_frequency_12m": 0.0, "total_claim_amount_12m": 0.0,
          "avg_claim_amount_12m": 0.0, "repeat_submission_flag": 0, "submission_delay_days": 23,
          "late_submission_flag": 0, "funeral_policy_flag": 0, "high_amount_flag": 0, "fraud_flag": 0,
          "loss_ratio": 0.0, "pricing_adequacy_flag": 1, "upgrade_flag": 1, "customer_lifetime_value": 127780.5
        },
        {
          "age": 49, "gender": "Female", "income_band": "Middle", "employment_status": "Self-Employed",
          "province": "KwaZulu-Natal", "urban_rural": "Urban", "household_size": 4, "province_risk_score": 0.18,
          "policy_type": "COID", "tenure_months": 189, "num_active_policies": 1, "sum_assured": 430400.68,
          "monthly_premium": 3155.87, "expected_annual_premium": 37870.44, "industry_type": "Agriculture",
          "num_employees": 36, "location": "Urban", "safety_score": 96, "coverage_type": "COID",
          "deductible": 5139.06, "policy_age_months": 189, "job_role": "Project Manager", "employee_age": 49,
          "health_score": 61, "risk_score": 55.0, "risk_level": "Medium", "past_claims_count": 0,
          "past_claims_amount": 0.0, "expected_loss": 0.0, "premium_recommendation": 0.0,
          "payment_method": "Debit Order", "debit_order_flag": 1, "missed_payments_12m": 0,
          "lapse_flag": 0, "claims_count_12m": 1, "claim_frequency_12m": 0.08,
          "total_claim_amount_12m": 21558.09, "avg_claim_amount_12m": 21558.09,
          "repeat_submission_flag": 0, "submission_delay_days": 30, "late_submission_flag": 0,
          "funeral_policy_flag": 0, "high_amount_flag": 0, "fraud_flag": 0, "loss_ratio": 0.57,
          "pricing_adequacy_flag": 1, "upgrade_flag": 0, "customer_lifetime_value": 174398.72
        }
      ]
    }
    
    body = str.encode(json.dumps(data))
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ auth_key), 'azureml-model-deployment': 'blue'}

    req = urllib.request.Request(scoring_uri, body, headers)

    print("\n🚀 Sending Request...")
    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        print(f"✅ Prediction: {result.decode('utf-8')}")
    except urllib.error.HTTPError as error:
        print(f"❌ Status Code: {error.code}")
        print(error.read().decode("utf8", 'ignore'))

if __name__ == "__main__":
    allowSelfSignedHttps(True)
    main()