import pytest
from sklearn.model_selection import train_test_split
from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.models.pipeline import build_and_train_pipeline
from src.models.evaluate import evaluate

config = {
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 10,  # small number for fast testing
    "max_depth": 3,
    "min_samples_split": 2,
    "class_weight": "balanced"
}

def test_train():
    df = load_data("data/credit_risk_dataset.csv")
    X, y = preprocess(df, "loan_status")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"]
    )

    pipeline = build_and_train_pipeline(X_train, y_train, config)

    # check pipeline exists
    assert pipeline is not None

    # check test set has rows
    assert X_test.shape[0] > 0

def test_evaluate():
    df = load_data("data/credit_risk_dataset.csv")
    X, y = preprocess(df, "loan_status")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"]
    )

    pipeline = build_and_train_pipeline(X_train, y_train, config)
    metrics = evaluate(pipeline, X_test, y_test)

    # check metrics exist
    assert "accuracy" in metrics
    assert "classification_report" in metrics

    # check accuracy is reasonable
    assert metrics["accuracy"] > 0.7
