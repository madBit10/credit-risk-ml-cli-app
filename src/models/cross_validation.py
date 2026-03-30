import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_cross_validation(X,y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=3, random_state=42, eval_metrics="logloss"),
        # "SVM": SVC(kernel="linear", C=0.1, gamma="scale", class_weight="balanced", probability=True)
        "NaiveBayes": GaussianNB()

    }

    results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="recall")
        results[name] = {
            "mean": round(scores.mean(), 4),
            "std": round(scores.std(), 4),
            "scores": [round(s,4) for s in scores]
        }
        logger.info(f"{name}: Recall = {scores.mean(): .4f} ± {scores.std(): .4f}")

    return results

def print_cv_results(results):
    logger.info("\n ===== Cross Validation Results (Recall) =====")
    for name, result in results.items():
        logger.info(f"{name}")
        logger.info(f"Fold score: {result['scores']}")
        logger.info(f"Mean Recall: {result['mean']}")
        logger.info(f"Std Dev: {result['std']}")
    logger.info("=================================================")