import os
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

def main():
    # 1. Connect to the Workspace using GitHub Secrets (Environment Variables)
    # These match the secret names you provided in your GitHub setup
    print("Connecting to workspace...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
        resource_group_name="aawasthi-rg", 
        workspace_name="sample_test_workspace_praharsha"
    )
    print("Connected to Workspace via SDK v2!")

    # 2. Define the cloud environment using your local env.yaml
    # This matches your existing project structure
    my_job_env = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file="env.yaml",
        name="insurance-cloud-env"
    )

    # 3. Define the Training Job
    # FIX: Compute name updated to 'sample-ml-compute1' to prevent your previous error
    job = command(
        code="./",  
        command="python train.py",
        environment=my_job_env,
        compute="sample-ml-compute1", 
        display_name="insurance-churn-prediction-v2",
        experiment_name="Insurance_Churn_V2"
    )

    # 4. Submit the Job
    print("Submitting job to Azure ML...")
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted! View it here: {returned_job.studio_url}")

    # 5. Wait for the job to complete (Necessary for Auto-Registration)
    print("Waiting for training job to complete...")
    ml_client.jobs.stream(returned_job.name)

    # 6. Register the Model from the Job Output
    print("Registering the winning model...")
    model_path = f"azureml://jobs/{returned_job.name}/outputs/artifacts/paths/model/"

    run_model = Model(
        path=model_path,
        name="insurance-churn-model-gh",
        description="Model trained and registered via GitHub Actions",
        type=AssetTypes.MLFLOW_MODEL
    )

    ml_client.models.create_or_update(run_model)
    print(f"Model successfully registered as: {run_model.name}")

if __name__ == "__main__":
    main()