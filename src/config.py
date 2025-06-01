import os

# MLflow settings
MLFLOW_TRACKING_URI = "mlruns"  # Local directory for MLflow tracking
EXPERIMENT_NAME = "xception_classifier"

# Model parameters
NUM_CLASSES = 4
IMG_SIZE = (299, 299, 3)
LEARNING_RATE = 0.001
FREEZE_BASE = False

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2

# Data parameters
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRAIN_DIR = os.path.join(DATA_DIR, "training")
TEST_DIR = os.path.join(DATA_DIR, "testing")

# Initialize MLflow
def setup_mlflow():
    """Initialize MLflow settings"""
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create experiment if it doesn't exist
    try:
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            mlflow.create_experiment(EXPERIMENT_NAME)
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception as e:
        print(f"Error setting up MLflow: {str(e)}")
        raise

# Model parameters
MODEL_NAME = "tumor_detection_model"
MODEL_VERSION = "1"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

# Ensure directories exist
os.makedirs(ARTIFACT_DIR, exist_ok=True) 
