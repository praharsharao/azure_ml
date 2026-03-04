import os
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
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

    # 1. Define the environment with a SAFE name (no periods)
    # We use the standard Azure ML base image and let env.yaml handle the rest.
    my_job_env = Environment(
        name="sklearn_xgboost_env_v5", # Use underscores, NO periods
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file="env.yaml"
    )

    job = command(
        code="./",  
        command="python train.py",
        environment=my_job_env,
        compute="sample-cluster-compute", 
        display_name="insurance-churn-final-v5",
        experiment_name="Insurance_Churn_V2"
    )

    print("Submitting job with fixed environment name...")
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted! View here: {returned_job.studio_url}")

if __name__ == "__main__":
    main()