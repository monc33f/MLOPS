import os
import sys
import requests
from pathlib import Path

def download_model(url, save_path):
    """
    Download the model file from a remote storage
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading model to {save_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print("Model downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

if __name__ == "__main__":
    # The URL should be provided as an environment variable or command line argument
    model_url = os.getenv("MODEL_URL") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not model_url:
        print("Please provide MODEL_URL environment variable or pass URL as argument")
        sys.exit(1)
        
    save_path = "saved_models/xception_model.keras"
    if not download_model(model_url, save_path):
        sys.exit(1) 