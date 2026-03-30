import numpy as np 
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)

def tune_forest(X,y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = {
        "n_estimators": [50,100,200],
        "max_depth": [3,5,7],
        "min_samples_split": [2,5,10]
    }

    model = RandomForestClassifier(class_weight="balanced", random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='recall',
        n_jobs=-1,
        verbose=1
    )

    logger.info("Starting GridSeachCV... this may take a few minutes")
    grid_search.fit(X,y)

    logger.info(f"Best params: {grid_search.best_params_}")
    logger.info(f"Best recall: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_