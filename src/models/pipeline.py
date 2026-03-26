from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from src.utils.logger import get_logger

logger = get_logger(__name__)

def build_and_train_pipeline(model_type, X_train, y_train, params: dict):

    logger.info("Starting the build and train pipeline...")

    if model_type == "rf":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                random_state=params["random_state"],
                class_weight=params["class_weight"]
            ))
        ]).fit(X_train, y_train)
    elif model_type == "svm":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel=params["kernel"],
                class_weight=params["class_weight"],
                random_state=params["random_state"],
                probability=params["probability"]
            ))
        ]).fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    logger.info("The build and train pipeline executed.....")
    
    return pipeline
