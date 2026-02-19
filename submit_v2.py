from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential

# 1. Connect to the Workspace (SDK v2 Style)
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="fad1a96b-a67e-452d-87d8-fcdb0a781616",
    resource_group_name="sample_uc",
    workspace_name="sample_test_uc1"
)

print("Connected to Workspace via SDK v2!")

# 2. Define the Job (The "Command")
# This tells Azure: "Run train.py on a cluster using this environment"
job = command(
    code="./",  # Upload all files in current folder
    command="python train.py",
    environment="azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest", # We use a pre-made Azure env for now
    compute="sample-cluster-compute", 
    display_name="insurance-churn-prediction-v2",
    experiment_name="Insurance_Churn_V2"
)

# 3. Submit the Job
print("Submitting job...")
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted! View it here: {returned_job.studio_url}")
