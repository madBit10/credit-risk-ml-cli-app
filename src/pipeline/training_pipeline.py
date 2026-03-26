# It orchestrates ("manages") the entire traning process

"""
calls:

step1 - load_data.py "Gets the raw data"
step2 - preprocess.py "Clean and prepare it"
step3 - train.py "Train the model"
step4 - evaluate.py "Check how good it is"
step5 - save_artifacts.py "Save the model and results"

"""

import yaml
from src.data.load_data import load_data
from src.data.preprocess import preprocess
# from src.models.train import train
from src.models.evaluate import evaluate
from src.utils.save_artifacts import save_artifacts
from src.utils.logger import get_logger
from src.models.cross_validation import run_cross_validation, print_cv_results
from src.models.tune import tune_forest
from sklearn.model_selection import train_test_split
from src.models.pipeline import build_and_train_pipeline

logger = get_logger(__name__)

def run_training_pipeline():
    """
    Orchestrates the full training process.
    Calls each step in order and passes data between them.


    """

    logger.info("Starting training pipeline...")

    # step1 - load data

    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded")

    # Step 2 - Load data
    df = load_data(config["data"]["raw_data_path"])

    # step 3 - Preprocess

    X,y = preprocess(df, config["target_column"])

    # split into train and test

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )
    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # step 4 - cross validation

    logger.info("Running Cross validation")
    cv_results = run_cross_validation(X,y)
    print_cv_results(cv_results)

    # step 5 - Hyperparameter Tuning 

    logger.info("Running hyperparameter tuning...")
    best_model, best_params, best_score = tune_forest(X,y)
    logger.info(f"Best model ready. Recall: {best_score:.4f}, Params: {best_params}")

    # Step 5 - Train

    pipeline = build_and_train_pipeline("rf", X_train,y_train, {
        # "test_size": config["data"]["test_size"],
        "n_estimators": best_params["n_estimators"],
        "max_depth": best_params["max_depth"],
        "min_samples_split": best_params["min_samples_split"],
        "random_state": config["data"]["random_state"],
        "class_weight": config["model"]["class_weight"]
    })

    # step 5' - Train SVM model

    svm_pipeline = build_and_train_pipeline("svm", X_train, y_train, {
        "kernel": config["svm"]["kernel"],
        "class_weight": config["svm"]["class_weight"],
        "random_state": config["svm"]["random_state"],
        "probability": config["svm"]["probability"]
    })

    # step 6 - Evaluate 

    rf_metrics = evaluate(pipeline, X_test, y_test)

    # step 6' - evaluate svm 

    svm_metrics = evaluate(svm_pipeline, X_test, y_test)

    # comparing 2 models

    # Compare and pick best model
    if rf_metrics['classification_report']['1']['recall'] > svm_metrics['classification_report']['1']['recall']:                                                                                                                     
        best_pipeline = pipeline                                                                                                                                   
        best_metrics = rf_metrics
        logger.info("Best model: RF")                                                                                                                              
    else:                                                                                                                                                          
        best_pipeline = svm_pipeline
        best_metrics = svm_metrics                                                                                                                                 
        logger.info("Best model: SVM")

    # comparison
    print("\n--- Model Comparison ---")
    print(f"RF  → Accuracy: {rf_metrics['accuracy']:.4f} | Recall: {rf_metrics['classification_report']['1']['recall']:.4f} | Precision: {rf_metrics['classification_report']['1']['precision']:.4f}")
    print(f"SVM → Accuracy: {svm_metrics['accuracy']:.4f} | Recall: {svm_metrics['classification_report']['1']['recall']:.4f} | Precision: {svm_metrics['classification_report']['1']['precision']:.4f}")

    # step 7 - Save artifacts

    save_artifacts(best_pipeline,X.columns.to_list(), best_metrics, {
        "model_path": config["artifacts"]["model_path"],
        # "scaler_path": config["artifacts"]["scaler_path"],
        "features_path": config["artifacts"]["features_path"],
        "metrics_path": config["artifacts"]["metrics_path"]
    })

    logger.info("Training pipeline complete!")