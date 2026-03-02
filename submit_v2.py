import os
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

def main():
    print("Connecting to workspace...")
    credential = DefaultAzureCredential()
    
    # Using your verified subscription and workspace from previous successful logs
    ml_client = MLClient(
        credential=credential,
        subscription_id="fad1a96b-a67e-452d-87d8-fcdb0a781616",
        resource_group_name="sample_uc",
        workspace_name="sample_test_uc1"
    )
    print("Connected to Workspace via SDK v2!")

    my_job_env = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file="env.yaml",
        name="insurance-cloud-env"
    )

    job = command(
        code="./",  
        command="python train.py",
        environment=my_job_env,
        compute="sample-cluster-compute", 
        display_name="insurance-churn-prediction-v2",
        experiment_name="Insurance_Churn_V2"
    )

    print("Submitting job...")
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted! View it here: {returned_job.studio_url}")

if __name__ == "__main__":
    main()
