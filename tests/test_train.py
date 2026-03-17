import pytest
from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.models.train import train
from src.models.evaluate import evaluate

def test_train():
    df = load_data("data/credit_risk_dataset.csv")
    X,y, scaler = preprocess(df, "loan_status")

    config = {
        "test_size": 0.2,                                                            
        "random_state": 42,                                                            
        "n_estimators": 10,  # small number for fast testing
        "max_depth": 3,                                                                
        "class_weight": "balanced"
    }

    model, X_test, y_test = train(X,y, config)

    # check model exists
    assert model is not None

    # check test set has rows
    assert X_test.shape[0] > 0

def test_evaluate():
    df = load_data("data/credit_risk_dataset.csv")
    X,y, scaler = preprocess(df, "loan_status")

    config = {
        "test_size": 0.2,                                                            
        "random_state": 42,                                                            
        "n_estimators": 10,  # small number for fast testing
        "max_depth": 3,                                                                
        "class_weight": "balanced"
    }

    model, X_test, y_test = train(X,y, config)
    metrics = evaluate(model, X_test, y_test)

    # check metrics exist 

    assert "accuracy" in metrics
    assert "classification_report" in metrics

    # check accuracy is reasonable
    assert metrics["accuracy"] > 0.7