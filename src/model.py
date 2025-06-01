import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.applications import Xception

def create_xception_model(num_classes=4, 
                         img_size=(299, 299, 3), 
                         learning_rate=0.001,
                         freeze_base=False):
    """
    Creates an Xception-based model with custom classification head.
    
    Args:
        num_classes: Number of output classes
        img_size: Input image dimensions (height, width, channels)
        learning_rate: Initial learning rate for Adamax optimizer
        freeze_base: Whether to freeze Xception base layers
    
    Returns:
        Compiled Keras model ready for training
    """
    # 1. Base Model (Xception)
    base_model = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=img_size,
        pooling='max'  # GlobalMaxPooling2D output
    )
    
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
    
    # 2. Custom Classification Head
    model = Sequential([
        base_model,
        Dropout(0.3),  # Additional dropout after pooling
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(num_classes, activation='softmax')
    ], name='xception_classifier')
    

    
    return model

