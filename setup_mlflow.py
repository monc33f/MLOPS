from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer

# Initialize ZenML client
client = Client()

# Register MLflow experiment tracker
mlflow_tracker = MLFlowExperimentTracker(
    name="mlflow_tracker",
    tracking_uri=get_tracking_uri(),
)
client.active_stack.experiment_tracker = mlflow_tracker

# Register MLflow model deployer
model_deployer = MLFlowModelDeployer(name="mlflow_deployer")
client.active_stack.model_deployer = model_deployer

print("MLflow integration configured successfully!")
print(f"MLflow tracking URI: {get_tracking_uri()}")
print("\nTo start the MLflow UI, run:")
print("mlflow ui") 