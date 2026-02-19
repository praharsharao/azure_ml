from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

# 1. Connect to the Workspace
print("Connecting to workspace...")
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="fad1a96b-a67e-452d-87d8-fcdb0a781616",
    resource_group_name="sample_uc",
    workspace_name="sample_test_uc1"
)
print("Connected to Workspace via SDK v2!")

# 2. Define the cloud environment using your local env.yaml
my_job_env = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file="env.yaml",
    name="insurance-cloud-env"
)

# 3. Define the Job
job = command(
    code="./",  
    command="python train.py",
    environment=my_job_env,
    compute="cpu-cluster", 
    display_name="insurance-churn-prediction-v2",
    experiment_name="Insurance_Churn_V2"
)

# 4. Submit the Job
print("Submitting job...")
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted! View it here: {returned_job.studio_url}")