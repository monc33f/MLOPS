from zenml import step
import logging
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import mlflow
import mlflow.keras
from src.config import setup_mlflow

@step
def train_model(
    tr_df: pd.DataFrame,
    valid_df: pd.DataFrame
) -> tf.keras.Model:
    # Set up MLflow
    setup_mlflow()
    
    with mlflow.start_run(run_name="model_training"):
        logging.info("Recreating ImageDataGenerators...")
        batch_size = 32
        img_size = (299, 299)
        
        # Log parameters
        mlflow.log_params({
            "batch_size": batch_size,
            "img_size": img_size,
            "learning_rate": 0.001,
            "optimizer": "Adamax"
        })

        train_datagen = ImageDataGenerator(rescale=1/255, brightness_range=(0.8, 1.2))
        valid_datagen = ImageDataGenerator(rescale=1/255)

        tr_gen = train_datagen.flow_from_dataframe(
            tr_df,
            x_col='Class Path',
            y_col='Class',
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

        valid_gen = valid_datagen.flow_from_dataframe(
            valid_df,
            x_col='Class Path',
            y_col='Class',
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

        logging.info("Building the Xception-based model...")
        base_model = Xception(
            include_top=False,
            weights="imagenet",
            input_shape=(299, 299, 3),
            pooling='max'
        )

        model = Sequential([
            base_model,
            Flatten(),
            Dropout(rate=0.3),
            Dense(128, activation='relu'),
            Dropout(rate=0.25),
            Dense(4, activation='softmax')
        ])

        model.compile(
            optimizer=Adamax(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )

        # Create MLflow callback
        class MLflowCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                for metric, value in logs.items():
                    mlflow.log_metric(metric, value, step=epoch)

        logging.info("Starting model training...")
        history = model.fit(
            tr_gen,
            validation_data=valid_gen,
            epochs=5,
            callbacks=[MLflowCallback()]
        )

        # Log final metrics
        for metric in history.history:
            mlflow.log_metric(f"final_{metric}", history.history[metric][-1])

        logging.info("Model training complete.")

        # Save model both locally and to MLflow
        save_dir = "saved_models"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save locally
        model_path = os.path.join(save_dir, "xception_model.keras")
        model.save(model_path)
        logging.info(f"Model saved locally to {model_path}")
        
        # Save to MLflow
        mlflow.keras.log_model(model, "model")
        logging.info("Model saved to MLflow")
        
        # Log the model path as an artifact
        mlflow.log_artifact(model_path, "saved_model")
        
        return model
