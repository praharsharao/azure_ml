
import datetime
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration, Environment
from azure.identity import DefaultAzureCredential

def main():
    ml_client = MLClient.from_config(DefaultAzureCredential())
    
    timestamp = datetime.datetime.now().strftime("%m%d%H%M")
    endpoint_name = f"insurance-v2-endpoint-{timestamp}"
    
    print(f"🚀 Creating Endpoint: {endpoint_name}")
    
    # 1. Create Endpoint
    endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")
    ml_client.begin_create_or_update(endpoint).wait()
    
    # 2. Deploy Model
    print("Deploying Model (approx 10-15 mins)...")
    deployment = ManagedOnlineDeployment(
        name="production",
        endpoint_name=endpoint_name,
        model="insurance-churn-v2:1",
        environment=Environment(
            conda_file="conda.yaml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
        ),
        code_configuration=CodeConfiguration(code="./", scoring_script="score.py"),
        # We use DS2 (2 cores) now that we cleaned up the quota
        instance_type="Standard_DS2_v2", 
        instance_count=1
    )
    
    ml_client.begin_create_or_update(deployment).wait()
    
    # 3. Set Traffic
    endpoint.traffic = {"production": 100}
    ml_client.begin_create_or_update(endpoint).wait()
    
    print(f"✅ Live at: {ml_client.online_endpoints.get(endpoint_name).scoring_uri}")

if __name__ == "__main__":
    main()