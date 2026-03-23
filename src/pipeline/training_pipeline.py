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
from src.models.train import train
from src.models.evaluate import evaluate
from src.utils.save_artifacts import save_artifacts
from src.utils.logger import get_logger
from src.models.cross_validation import run_cross_validation, print_cv_results
from src.models.tune import tune_forest

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

    X,y,scaler = preprocess(df, config["target_column"])

    # step 4 - cross validation

    logger.info("Running Cross validation")
    cv_results = run_cross_validation(X,y)
    print_cv_results(cv_results)

    # step 5 - Hyperparameter Tuning 

    logger.info("Running hyperparameter tuning...")
    best_model, best_params, best_score = tune_forest(X,y)
    logger.info(f"Best model ready. Recall: {best_score:.4f}, Params: {best_params}")

    # Step 5 - Train

    model, X_test, y_test = train(X,y, {
        "test_size": config["data"]["test_size"],
        "random_state": config["data"]["random_state"],
        "n_estimators": best_params["max_depth"],
        "max_depth": best_params["max_depth"],
        "min_samples_split": best_params["min_samples_split"],
        "class_weight": config["model"]["class_weight"]
    })

    # step 6 - Evaluate 

    metrics = evaluate(model, X_test, y_test)

    # step 7 - Save artifacts

    save_artifacts(model, scaler,X.columns.to_list(), metrics, {
        "model_path": config["artifacts"]["model_path"],
        "scaler_path": config["artifacts"]["scaler_path"],
        "features_path": config["artifacts"]["features_path"],
        "metrics_path": config["artifacts"]["metrics_path"]
    })

    logger.info("Training pipeline complete!")