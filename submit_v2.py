import os
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

def main():
    print("Connecting to workspace...")
    credential = DefaultAzureCredential()
    
    # Authenticate to your specific workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id="fad1a96b-a67e-452d-87d8-fcdb0a781616",
        resource_group_name="sample_uc",
        workspace_name="sample_test_uc1"
    )
    print("Connected successfully!")

    # Using the curated environment from your image_00fcca.png
    # We use version 39 as shown in your screenshot.
    curated_env = "azureml://registries/azureml/environments/sklearn-1.5/versions/39"

    # Define the command job
    job = command(
        code="./",  # Path to your train.py
        command="python train.py",
        environment=curated_env,
        compute="sample-cluster-compute", 
        display_name="insurance-churn-final-run",
        experiment_name="Insurance_Churn_V2"
    )

    print("Submitting job using Curated Environment...")
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted! View status here: {returned_job.studio_url}")

if __name__ == "__main__":
    main()