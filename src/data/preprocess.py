# Data processing layer

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

def preprocess(df: pd.DataFrame, target_column: str):
    """
    Cleans and preprocess raw data for training

    Args:
        df: Raw Dataframe
        target_column: Name of the target column
    
    Returns: 
        X: input features
        y: target columns

    """

    logger.info("Starting preprocessing...")


    # step 1 - Drop duplicates

    df = df.drop_duplicates().copy()
    logger.info(f"After dropping the duplicates: {df.shape}")


    # step2 - Fill missing values with median 

    # Fill the loan int rate with median missing values handling
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

    # fill person emp_length with median handling missing values 

    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())

    logger.info("Missing values filled")


    # step 3 - Convert Y/N to 1/0

    # convert Y/N to 1/0

    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map({"Y": 1, "N":0})


    # step 4 - Label encode loan_grade column

    # using label encoder on the loan grade column as it has the data that follows a specific order

    le = LabelEncoder() 
    df["loan_grade"] = le.fit_transform(df["loan_grade"])

    logger.info("loan_grade label encoded")


    # Step 5 - one hot label encoding to the columns 

    # let's change the home_ownership and loan_intent columns through one hot encoding

    df = pd.get_dummies(df, columns=["person_home_ownership", "loan_intent"], drop_first=True)

    logger.info("Categorical columns one hot encoded")


    # step 6 - fix boolean columns to int

    df = df.astype({col: int for col in df.select_dtypes(include="bool").columns})


    # step 7 - Split X and y 

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # # step 8 - scale numeric values 

    # numeric_cols = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]

    # scaler = StandardScaler()
    # X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    # logger.info("Features scaling complete")

    logger.info(f"Preprocessing complete. X shape: {X.shape}, y shape: {y.shape}")

    return X,y

