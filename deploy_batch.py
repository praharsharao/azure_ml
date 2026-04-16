from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    BatchEndpoint,
    ModelBatchDeployment,
    BatchRetrySettings,
    CodeConfiguration,
    Environment 
)
from azure.identity import DefaultAzureCredential

def main():
    print(" Authenticating with Azure...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id="fad1a96b-a67e-452d-87d8-fcdb0a781616",
        resource_group_name="sample_uc",
        workspace_name="sample_test_uc1"
    )

    endpoint_name = "insurance-churn-batch-endpoint"


    print(f" Verifying Batch Endpoint: {endpoint_name}...")
    endpoint = BatchEndpoint(
        name=endpoint_name,
        description="Batch processing for monthly insurance churn data"
    )
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()

    print(" Fetching LATEST model version...")
    latest_model = ml_client.models.get(name="insurance-churn-prediction-model", label="latest")
    

    print(" Building clean, explicitly defined environment (v5 Cache Break)...")
    clean_env = Environment(
        name="insurance-batch-clean-v5", 
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        conda_file="./src/conda.yaml" 
    )

   
    print(" Deploying the model to the batch endpoint...")
    deployment = ModelBatchDeployment(
        name="insurance-batch-dp",
        endpoint_name=endpoint_name,
        model=latest_model, 
        environment=clean_env,
        code_configuration=CodeConfiguration(
            code="./src", 
            scoring_script="score_batch.py"
        ),
        compute="sample-cluster-compute",
        instance_count=1,
        max_concurrency_per_instance=2,
        mini_batch_size=10,
        retry_settings=BatchRetrySettings(max_retries=3, timeout=30),
        output_file_name="batch_predictions.csv"
    )
    
    ml_client.batch_deployments.begin_create_or_update(deployment).result()

   
    endpoint.defaults = {"deployment_name": deployment.name}
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    
    print(" Batch Deployment Complete!")

if __name__ == "__main__":
    main()