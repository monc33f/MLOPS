from typing import Tuple
import pandas as pd
from zenml import step
from src.get_data import train_df, test_df
from sklearn.model_selection import train_test_split
import mlflow

@step
def ingest_df(data_path1: str, data_path2: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with mlflow.start_run(run_name="data_ingestion"):
        train_data = train_df(data_path1)
        test_data = test_df(data_path2)

        # Log dataset sizes
        mlflow.log_params({
            "total_train_samples": len(train_data),
            "test_samples": len(test_data),
            "train_path": data_path1,
            "test_path": data_path2
        })

        # Split training data
        train_split, valid_split = train_test_split(train_data, test_size=0.2, random_state=42)
        
        # Log split sizes
        mlflow.log_params({
            "train_split_samples": len(train_split),
            "validation_split_samples": len(valid_split),
            "validation_split_ratio": 0.2
        })

        # Log class distribution
        train_class_dist = train_split['Class'].value_counts().to_dict()
        valid_class_dist = valid_split['Class'].value_counts().to_dict()
        test_class_dist = test_data['Class'].value_counts().to_dict()
        
        mlflow.log_params({
            "train_class_distribution": train_class_dist,
            "validation_class_distribution": valid_class_dist,
            "test_class_distribution": test_class_dist
        })

        return train_split, valid_split, test_data
