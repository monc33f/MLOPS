import logging
from typing import Dict, List
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import DirectoryIterator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model: Model, 
                 train_gen: DirectoryIterator, 
                 valid_gen: DirectoryIterator, 
                 test_gen: DirectoryIterator, 
                 metrics: List[str] = ['loss', 'accuracy']) -> Dict:
    """
    Evaluates model performance on training, validation, and test sets with detailed logging.
    
    Args:
        model: Compiled Keras model
        train_gen: Training data generator
        valid_gen: Validation data generator
        test_gen: Test data generator
        metrics: List of metrics to evaluate (must match model metrics)
        
    Returns:
        Dictionary containing all evaluation results
        
    Example:
        results = evaluate_model(model, tr_gen, valid_gen, ts_gen)
    """
    logger.info("Starting model evaluation...")
    logger.debug(f"Evaluating with metrics: {metrics}")
    
    # Validate metrics
    available_metrics = model.metrics_names
    for metric in metrics:
        if metric not in available_metrics:
            error_msg = f"Metric '{metric}' not found in model metrics. Available: {available_metrics}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Evaluate on all datasets
    results = {}
    try:
        logger.info("Evaluating on training set...")
        results['train'] = model.evaluate(train_gen, verbose=0, return_dict=True)
        logger.debug(f"Training results: {results['train']}")
        
        logger.info("Evaluating on validation set...")
        results['valid'] = model.evaluate(valid_gen, verbose=0, return_dict=True)
        logger.debug(f"Validation results: {results['valid']}")
        
        logger.info("Evaluating on test set...")
        results['test'] = model.evaluate(test_gen, verbose=0, return_dict=True)
        logger.debug(f"Test results: {results['test']}")
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise
    
    # Log formatted results
    logger.info("\nMODEL PERFORMANCE EVALUATION")
    logger.info("=" * 50)
    
    for dataset in ['train', 'valid', 'test']:
        logger.info(f"\n{dataset.upper()} SET RESULTS:")
        logger.info("-" * 30)
        for metric in metrics:
            value = results[dataset][metric]
            if metric == 'accuracy':
                logger.info(f"{metric.capitalize()}: {value*100:.2f}%")
            else:
                logger.info(f"{metric.capitalize()}: {value:.4f}")
    
    logger.info("\n" + "=" * 50)
    
    logger.info("Model evaluation completed successfully.")
    return results