import os
import pandas as pd
from typing import List, Tuple

def train_df(tr_path: str) -> pd.DataFrame:
    """
    Create a DataFrame of image paths and their corresponding class labels from a directory structure.
    
    Args:
        tr_path (str): Path to the training directory containing class subdirectories
        
    Returns:
        pd.DataFrame: DataFrame with columns 'Class Path' and 'Class'
    
    Example Directory Structure:
        tr_path/
        ├── class1/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── class2/
            ├── image1.jpg
            └── image2.jpg
    """
    # Validate input path exists
    if not os.path.exists(tr_path):
        raise ValueError(f"Training directory not found: {tr_path}")
    
    # Get all (class, image_path) pairs
    classes, class_paths = zip(*[
        (label, os.path.join(tr_path, label, image))
        for label in os.listdir(tr_path) 
        if os.path.isdir(os.path.join(tr_path, label))  # Only process directories
        for image in os.listdir(os.path.join(tr_path, label))
        if image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))  # Only image files
    ])
    
    # Create DataFrame
    tr_df = pd.DataFrame({
        'Class Path': class_paths,
        'Class': classes
    })
    
    # Shuffle the DataFrame
    tr_df = tr_df.sample(frac=1).reset_index(drop=True)
    
    return tr_df


def test_df(ts_path):
    classes, class_paths = zip(*[(label, os.path.join(ts_path, label, image))
                                 for label in os.listdir(ts_path) if os.path.isdir(os.path.join(ts_path, label))
                                 for image in os.listdir(os.path.join(ts_path, label))])

    ts_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return ts_df

def get_data(data_path) : 
    L =   [1,2,3]
    df = pd.DataFrame(L)
    return df