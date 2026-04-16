import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
    Environment
)
from azure.identity import DefaultAzureCredential

def main():
    print("Connecting to workspace...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id="fad1a96b-a67e-452d-87d8-fcdb0a781616",
        resource_group_name="sample_uc",
        workspace_name="sample_test_uc1"
    )

    endpoint_name = "insurance-churn-live-api" 
    
    print(f"Checking/Creating endpoint: {endpoint_name}...")
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Live REST API for Insurance Churn Predictions",
        auth_mode="key"
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()


    my_custom_env = Environment(
        name="insurance-custom-env-v9", 
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file="env.yaml"
    )

    print("Creating 'blue' deployment (This step provisions hardware and takes 5-10 minutes)...")
    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint_name,
        model="insurance-churn-prediction-model:1", 
        environment=my_custom_env, 
        code_configuration=CodeConfiguration(
            code="src", 
            scoring_script="score.py"
        ),
        instance_type="Standard_D2as_v4",
        instance_count=1,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    print("Routing 100% of live traffic to the 'blue' deployment...")
    endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    print("Deployment Successful!")
    print(f"Your Live API URL: {ml_client.online_endpoints.get(endpoint_name).scoring_uri}")

if __name__ == "__main__":
    main()