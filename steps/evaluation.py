from zenml import step
import logging
from typing import Dict
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

@step
def evaluate_model(
    model: tf.keras.Model,
    tr_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    ts_df: pd.DataFrame
) -> Dict[str, float]:
    
    with mlflow.start_run(run_name="model_evaluation"):
        logging.info("Recreating data generators for evaluation...")
        img_size = (299, 299)
        batch_size = 32

        eval_datagen = ImageDataGenerator(rescale=1/255)

        tr_gen = eval_datagen.flow_from_dataframe(
            tr_df,
            x_col='Class Path',
            y_col='Class',
            target_size=img_size,
            batch_size=batch_size,
            shuffle=False,
            class_mode='categorical'
        )

        valid_gen = eval_datagen.flow_from_dataframe(
            valid_df,
            x_col='Class Path',
            y_col='Class',
            target_size=img_size,
            batch_size=batch_size,
            shuffle=False,
            class_mode='categorical'
        )

        ts_gen = eval_datagen.flow_from_dataframe(
            ts_df,
            x_col='Class Path',
            y_col='Class',
            target_size=img_size,
            batch_size=16,
            shuffle=False,
            class_mode='categorical'
        )

        logging.info("Evaluating model on training data...")
        train_score = model.evaluate(tr_gen, verbose=1)
        mlflow.log_metrics({
            "train_loss": train_score[0],
            "train_accuracy": train_score[1],
            "train_precision": train_score[2],
            "train_recall": train_score[3]
        })

        logging.info("Evaluating model on validation data...")
        valid_score = model.evaluate(valid_gen, verbose=1)
        mlflow.log_metrics({
            "val_loss": valid_score[0],
            "val_accuracy": valid_score[1],
            "val_precision": valid_score[2],
            "val_recall": valid_score[3]
        })

        logging.info("Evaluating model on test data...")
        test_score = model.evaluate(ts_gen, verbose=1)
        mlflow.log_metrics({
            "test_loss": test_score[0],
            "test_accuracy": test_score[1],
            "test_precision": test_score[2],
            "test_recall": test_score[3]
        })

        # Generate and log confusion matrix
        y_pred = model.predict(ts_gen)
        y_true = ts_gen.classes
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save confusion matrix
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()

        logging.info(f"Train Loss: {train_score[0]:.4f}")
        logging.info(f"Train Accuracy: {train_score[1]*100:.2f}%")
        logging.info('-' * 20)
        logging.info(f"Validation Loss: {valid_score[0]:.4f}")
        logging.info(f"Validation Accuracy: {valid_score[1]*100:.2f}%")
        logging.info('-' * 20)
        logging.info(f"Test Loss: {test_score[0]:.4f}")
        logging.info(f"Test Accuracy: {test_score[1]*100:.2f}%")

        results = {
            "train_loss": train_score[0],
            "train_accuracy": train_score[1],
            "val_loss": valid_score[0],
            "val_accuracy": valid_score[1],
            "test_loss": test_score[0],
            "test_accuracy": test_score[1]
        }

        return results
