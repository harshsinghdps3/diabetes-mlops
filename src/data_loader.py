import pandas as pd
import numpy as np
import yaml

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(path):
    return pd.read_csv(path)

def preprocess_zeros(df, cols_with_zeros):
    """Replace 0s with NaN in specific columns for imputation later."""
    df_copy = df.copy()
    df_copy[cols_with_zeros] = df_copy[cols_with_zeros].replace(0, np.nan)
    return df_copy

if __name__ == "__main__":
    # Simple test run
    cfg = load_config()
    # Mocking data load if file exists, else print instruction
    print(f"Config loaded. Raw data path: {cfg['data']['raw_path']}")