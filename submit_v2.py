# Update this section inside your main() function:
    my_custom_env = Environment(
        name="insurance-custom-env-v8", # Incremented to v8
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file="env.yaml"
    )
    
    job = command(
        code="./",  
        command="python train.py",
        environment=my_custom_env,
        compute="sample-cluster-compute", 
        display_name="insurance-churn-v8-run",
        experiment_name="Insurance_Churn_V2"
    )