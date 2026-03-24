import pickle
import json
from src.utils.logger import get_logger

logger = get_logger(__name__)

def save_artifacts(model, features, metrics: dict, config: dict):
    """
    
    Saves trained model and metrics to disk

    Args: 
        pipeline: Trained sklearn pipeline
        metrics: Dictionary of evaluation metrics
        config: Artifacts path from config.yaml

    """

    # Save model

    model_path = config["model_path"]
    with open(model_path, "wb") as f:
        pickle.dump(model,f)
    logger.info(f"Model saved at {model_path}")

    # save scaler

    # scaler_path = config["scaler_path"]
    # with open(scaler_path, "wb") as f:
    #     pickle.dump(scaler, f)
    # logger.info(f"Scaler saved at {scaler_path}")

    # Save feature columns                                                                                                                                                                                   
    features_path = config["features_path"]                                                                                                                                                                
    with open(features_path, "w") as f:                                                                                                                                                                      
        json.dump(features, f)                                                                                                                                                                               
    logger.info(f"Features saved at {features_path}")

    # save_metrics

    metrics_path = config["metrics_path"]
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved at {metrics_path}")