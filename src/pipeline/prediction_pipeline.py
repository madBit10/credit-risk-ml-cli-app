import yaml
import pickle
import json
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_prediction_pipeline(input_path: str):
    """loads saved model and scaler, preprocess input data and returns predictions. 
    
    Args: 
        input_path: Path to the csv file with new data to predict on

    """

    logger.info("Starting prediction pipeline...")

    # step 1 - Load Config

    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded")

    # step 2 - load model

    with open(config["artifacts"]["model_path"], "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded")

    # step 3 - Load Scaler 
    # with open(config["artifacts"]["scaler_path"], "rb") as f:
    #     scaler = pickle.load(f)
    # logger.info("Scaler loaded")

    # Step 4 - Load input data 
    df = pd.read_csv(input_path)
    logger.info(f"Input data loaded. Shape: {df.shape}")

    # Load features and reindex prediction data:
  # Load feature columns                                                                                                                                                                                       
    with open(config["artifacts"]["features_path"], "r") as f:                                                                                                                                                 
        feature_columns = json.load(f)                                                                                                                                                                           
    logger.info("Feature columns loaded")

    # step 5 - preprocess input
    df = preprocess_input(df, feature_columns)

    # step 6 - predict

    predictions = model.predict(df)

    probablities = model.predict_proba(df)[:, 1]

    # step 7 - show results 

    df["prediction"] = predictions
    df["default_probability"] = probablities.round(3)

    for i, (pred, prob) in enumerate(zip(predictions, probablities)):
        status = "WILL DEFAULT" if pred == 1 else "WILL NOT DEFAULT"
        logger.info(f"person {i+1}: {status} (probablility: {prob:.3f})")

    logger.info("Prediction pipeline complete!")

    return predictions, probablities


def preprocess_input(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Applies same preprocessing as traning but uses the saved scaler
    """

    from sklearn.preprocessing import LabelEncoder

    # convert Y/N to 1/0

    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map({"Y": 1, "N":0})

    # using label encoder on the loan grade column as it has the data that follows a specific order

    le = LabelEncoder() 
    df["loan_grade"] = le.fit_transform(df["loan_grade"])
    logger.info("loan_grade label encoded")

    # let's change the home_ownership and loan_intent columns through one hot encoding

    df = pd.get_dummies(df, columns=["person_home_ownership", "loan_intent"], drop_first=True)
    logger.info("Categorical columns one hot encoded")

    # step  - fix boolean columns to int
    df = df.astype({col: int for col in df.select_dtypes(include="bool").columns})

    # step - add missing columns with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # step 8 - reorder columns to match traning 

    df = df[feature_columns]

    # # Scale using saved scaler — transform only, NOT fit_transform
    # # Scale using saved scaler                                                                                                                                                                             
    # numeric_cols = ["person_age", "person_income", "person_emp_length",
    #                   "loan_amnt", "loan_int_rate", "loan_percent_income",                                                                                                                                     
    #                   "cb_person_cred_hist_length"]                                                                                                                                                            
    # # df[numeric_cols] = scaler.transform(df[numeric_cols])


    return df