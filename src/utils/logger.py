"""
Every other file will use the logger to print messages like:                         
  [INFO] Loading data...                                                                 
  [INFO] Preprocessing complete                                                        
  [ERROR] File not found

"""

import logging
import os


def get_logger(name: str) -> logging.Logger:

    """
    Creates and returns a logger.

    Args: 
        name: Name of the module using the logger

    Returns: 
        Configured logger instance
    
    """

    os.makedirs("artifacts/logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("artifacts/logs/training.log"),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(name)