# Data proecessing layer

import pandas as pd 
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_data(filepath: str) -> pd.DataFrame:

    """
    Loads raw CSV data from the given filepath.

    Args: 
        filepath: Path to the CSV file

    Returns:
        raw dataframe

    """

    logger.info(f"loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df