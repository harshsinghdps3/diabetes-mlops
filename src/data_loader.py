import pandas as pd
import numpy as np
import yaml
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise

def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        df = pd.read_csv(path)
        logger.info(f"Data loaded from {path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found at {path}")
        raise

def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """
    Validate that the DataFrame contains the required columns.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        required_cols (List[str]): List of column names that must be present.

    Returns:
        bool: True if validation passes.

    Raises:
        ValueError: If any required column is missing.
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.info("Column validation passed.")
    return True

def preprocess_zeros(df: pd.DataFrame, cols_with_zeros: List[str]) -> pd.DataFrame:
    """
    Replace 0s with NaN in specific columns for imputation later.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols_with_zeros (List[str]): List of columns where 0s should be replaced.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    df_copy = df.copy()
    for col in cols_with_zeros:
        if col in df_copy.columns:
            count = (df_copy[col] == 0).sum()
            df_copy[col] = df_copy[col].replace(0, np.nan)
            logger.info(f"Replaced {count} zeros with NaN in column '{col}'")
        else:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping zero replacement.")
    return df_copy

if __name__ == "__main__":
    # Simple test run
    try:
        cfg = load_config()
        # Mocking data load if file exists, else print instruction
        logger.info(f"Config loaded. Raw data path: {cfg['data']['raw_path']}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")