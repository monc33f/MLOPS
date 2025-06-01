import requests
import os
import pytest
import time

STREAMLIT_PORT = os.getenv("STREAMLIT_PORT", "8999")
MLFLOW_PORT = os.getenv("MLFLOW_PORT", "5000")

def test_streamlit_service():
    """Test if Streamlit service is running"""
    url = f"http://localhost:{STREAMLIT_PORT}"
    max_retries = 5
    retry_delay = 2

    for _ in range(max_retries):
        try:
            response = requests.get(url)
            assert response.status_code == 200
            return
        except (requests.ConnectionError, AssertionError):
            time.sleep(retry_delay)
    
    pytest.fail("Streamlit service is not responding")

def test_mlflow_service():
    """Test if MLflow service is running"""
    url = f"http://localhost:{MLFLOW_PORT}"
    max_retries = 5
    retry_delay = 2

    for _ in range(max_retries):
        try:
            response = requests.get(url)
            assert response.status_code == 200
            return
        except (requests.ConnectionError, AssertionError):
            time.sleep(retry_delay)
    
    pytest.fail("MLflow service is not responding")

def test_model_file_exists():
    """Test if the model file exists in the correct location"""
    model_path = "saved_models/xception_model.keras"
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    assert os.path.getsize(model_path) > 0, "Model file is empty" 