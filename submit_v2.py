import os
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

def main():
    # 1. Connect to the verified Workspace from your successful run
    print("Connecting to workspace...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.getenv("fad1a96b-a67e-452d-87d8-fcdb0a781616"),
        resource_group_name="sample_uc",      # Verified from your log
        workspace_name="sample_test_uc1"      # Verified from your log
    )
    print("Connected to Workspace via SDK v2!")

    # 2. Define the cloud environment
    my_job_env = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file="env.yaml",
        name="insurance-cloud-env"
    )

    # 3. Define the Training Job
    # Using your confirmed compute cluster name
    job = command(
        code="./",  
        command="python train.py",
        environment=my_job_env,
        compute="sample-ml-compute1", # Reverted to your successful cluster
        display_name="insurance-churn-prediction-v2",
        experiment_name="Insurance_Churn_V2"
    )

    # 4. Submit the Job
    print("Submitting job to Azure ML...")
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted! View it here: {returned_job.studio_url}")

    # 5. Wait and Register
    print("Waiting for training job to complete...")
    ml_client.jobs.stream(returned_job.name)

    print("Registering the winning model...")
    model_path = f"azureml://jobs/{returned_job.name}/outputs/artifacts/paths/model/"
    run_model = Model(
        path=model_path,
        name="insurance-churn-model-gh",
        description="Model registered via GitHub Actions",
        type=AssetTypes.MLFLOW_MODEL
    )
    ml_client.models.create_or_update(run_model)
    print(f"Model successfully registered: {run_model.name}")

if __name__ == "__main__":
    main()