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
    print("Connected successfully!")

    # Using a clean base with our strictly versioned env.yaml
    my_custom_env = Environment(
        name="insurance-custom-env-v7", # Incremented to v7
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file="env.yaml"
    )

    job = command(
        code="./",  
        command="python train.py",
        environment=my_custom_env,
        compute="sample-cluster-compute", 
        display_name="insurance-churn-v7-run",
        experiment_name="Insurance_Churn_V2"
    )

    print("Submitting job...")
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted! View status here: {returned_job.studio_url}")

if __name__ == "__main__":
    main()