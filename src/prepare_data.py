from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd

def create_data_generators(tr_df: pd.DataFrame, 
                          ts_df: pd.DataFrame,
                          img_size: tuple = (299, 299),
                          batch_size: int = 32,
                          validation_split: float = 0.5,
                          random_state: int = 20) -> tuple:
    """
    Creates train, validation, and test data generators with proper augmentation and splitting.
    
    Args:
        tr_df: DataFrame containing training data paths and labels
        ts_df: DataFrame containing test data paths and labels
        img_size: Target image dimensions (height, width)
        batch_size: Batch size for training/validation
        validation_split: Fraction of test data to use for validation
        random_state: Random seed for reproducible splits
        
    Returns:
        tuple: (train_gen, valid_gen, test_gen) - Keras DataGenerators
        
    Example Usage:
        tr_gen, valid_gen, ts_gen = create_data_generators(tr_df, ts_df)
    """
    # Split test data into validation and final test sets
    valid_df, ts_df = train_test_split(
        ts_df,
        train_size=validation_split,
        random_state=random_state,
        stratify=ts_df['Class']
    )
    
    # Configure data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        brightness_range=[0.8, 1.2],  # 20% brightness variation
        # Add more augmentations as needed:
        # rotation_range=20,
        # width_shift_range=0.2,
        # horizontal_flip=True
    )
    
    # Configuration for validation and test (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=tr_df,
        x_col='Class Path',
        y_col='Class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=random_state
    )
    
    valid_gen = train_datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col='Class Path',
        y_col='Class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=random_state
    )
    
    test_gen = test_datagen.flow_from_dataframe(
        dataframe=ts_df,
        x_col='Class Path',
        y_col='Class',
        target_size=img_size,
        batch_size=batch_size//2,  # Smaller batches for testing
        class_mode='categorical',
        shuffle=False  # Important for correct evaluation
    )
    
    # Print class indices for reference
    print("\nClass Indices:", train_gen.class_indices)
    
    return train_gen, valid_gen, test_gen