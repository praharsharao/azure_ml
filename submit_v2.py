from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment

# 1. Define the cloud environment using your local files
my_job_env = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file="env.yaml",
    name="insurance-cloud-env"
)

# 2. Tell the job to use it
job = command(
    code="./",  
    command="python train.py",
    environment=my_job_env,      # <--- Now it uses your requirements!
    compute="cpu-cluster", 
    display_name="insurance-churn-prediction-v2",
    experiment_name="Insurance_Churn_V2"
)
# 3. Submit the Job
print("Submitting job...")
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted! View it here: {returned_job.studio_url}")
