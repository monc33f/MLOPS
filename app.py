import streamlit as st
import mlflow
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from src.config import setup_mlflow, EXPERIMENT_NAME
import os
from datetime import datetime
import socket

# Set page config with Docker-specific settings
st.set_page_config(
    page_title="Medical Image Classification",
    page_icon="🏥",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

def load_latest_model():
    """Load the latest model from MLflow or local saved model"""
    try:
        # First try loading from local saved models
        local_model_path = os.path.join("saved_models", "xception_model.keras")
        if os.path.exists(local_model_path):
            model = tf.keras.models.load_model(local_model_path)
            st.sidebar.success("✅ Loaded model from local storage")
            # Create a structure similar to MLflow run for consistency
            local_run = {
                "is_local": True,
                "local_path": local_model_path,
                "timestamp": datetime.fromtimestamp(os.path.getmtime(local_model_path)).strftime("%Y-%m-%d %H:%M:%S")
            }
            return model, local_run
    except Exception as e:
        st.warning(f"Could not load local model: {str(e)}")

    # If local loading fails, try MLflow
    try:
        # Initialize MLflow
        setup_mlflow()
        
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        
        if experiment is None:
            available_experiments = client.search_experiments()
            if available_experiments:
                # Use the first available experiment
                experiment = available_experiments[0]
                st.warning(f"Using available experiment: {experiment.name}")
            else:
                st.error("No experiments found. Please train a model first.")
                return None
            
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"]
        )
        
        if not runs:
            st.error("No runs found. Please train a model first.")
            return None
            
        latest_run = runs[0]
        
        # Try loading as a keras model first
        try:
            model_uri = f"runs:/{latest_run.info.run_id}/model"
            model = mlflow.keras.load_model(model_uri)
            st.sidebar.success("✅ Loaded model from MLflow")
            return model, {"is_local": False, "run": latest_run}
        except Exception as e:
            st.warning(f"Could not load as Keras model: {str(e)}")
            
            # Try loading as a TensorFlow SavedModel
            try:
                artifact_path = client.download_artifacts(latest_run.info.run_id, "model")
                model = tf.keras.models.load_model(artifact_path)
                st.sidebar.success("✅ Loaded model from MLflow artifacts")
                return model, {"is_local": False, "run": latest_run}
            except Exception as e2:
                st.error(f"Could not load model from artifacts: {str(e2)}")
                return None
                
    except Exception as e:
        st.error(f"Error accessing MLflow: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image"""
    # Resize image to match model's expected input
    image = image.resize((299, 299))
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_class_label(prediction):
    """Convert model prediction to class label"""
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    return class_names[np.argmax(prediction)]

def main():
    st.title("🧠 Brain Tumor Classification")
    st.write("Upload a brain MRI image for tumor classification")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses a deep learning model to classify brain MRI images "
        "into four categories: Glioma, Meningioma, No Tumor, and Pituitary Tumor."
    )
    
    # Model loading
    model_load = load_latest_model()
    if model_load is None:
        st.stop()
    model, run_info = model_load
    
    # Display model info
    st.sidebar.title("Model Information")
    if run_info.get("is_local", False):
        st.sidebar.write("Model Source: Local Storage")
        st.sidebar.write(f"Model Path: {run_info['local_path']}")
        st.sidebar.write(f"Last Modified: {run_info['timestamp']}")
    else:
        mlflow_run = run_info["run"]
        st.sidebar.write(f"Model Source: MLflow")
        st.sidebar.write(f"Run ID: {mlflow_run.info.run_id[:8]}...")
        st.sidebar.write(f"Training Accuracy: {mlflow_run.data.metrics.get('train_accuracy', 'N/A'):.4f}")
        st.sidebar.write(f"Validation Accuracy: {mlflow_run.data.metrics.get('val_accuracy', 'N/A'):.4f}")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # Make prediction
        try:
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_class = get_class_label(prediction)
            confidence = float(np.max(prediction))
            
            with col2:
                st.subheader("Prediction Results")
                st.write("**Predicted Class:**", predicted_class)
                st.write("**Confidence Score:**", f"{confidence:.2%}")
                
                # Create a bar chart for all class probabilities
                st.write("**Probability Distribution:**")
                class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
                probabilities = prediction[0]
                
                # Create probability bars
                for class_name, prob in zip(class_names, probabilities):
                    st.write(f"{class_name}: {prob:.2%}")
                    st.progress(float(prob))
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Additional information
    st.markdown("---")
    st.write("""
    ### Instructions
    1. Upload a brain MRI image using the file uploader above
    2. The model will analyze the image and provide predictions
    3. View the predicted class and confidence scores
    
    ### Note
    - Supported image formats: JPG, JPEG, PNG
    - For best results, use clear MRI images
    - The model works best with properly oriented brain MRI scans
    """)

if __name__ == "__main__":
    main() 