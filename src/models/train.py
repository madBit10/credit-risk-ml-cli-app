import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train(X,y,config:dict):
    """
    Trains a Random Forest model

    Args:
        X: Input features
        y: Target column
        config: 

    Returns:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    """
    logger.info("Starting model training...")

    # split into train and test

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,
        test_size=config["test_size"],
        random_state=config["random_state"]
    )
    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # Train the model 

    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config.get("min_samples_split", 2),
        random_state=config["random_state"],
        class_weight=config["class_weight"]
    )

    model.fit(X_train, y_train)
    logger.info("Model training complete")

    return model, X_test, y_test