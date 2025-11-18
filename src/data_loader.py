import logging

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw diabetes dataset with validation."""
        raw_path = self.config["data"]["raw_path"]
        logger.info(f"Loading data from {raw_path}")

        df = pd.read_csv(raw_path)

        # Validate schema
        expected_cols = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "Outcome",
        ]
        assert all(col in df.columns for col in expected_cols), "Missing columns"
        assert df.shape[0] > 0, "Empty dataset"

        logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Perform data quality checks."""
        checks = []

        # Check for nulls
        null_count = df.isnull().sum().sum()
        checks.append(null_count == 0)
        logger.info(f"Null values: {null_count}")

        # Check target distribution
        target_col = self.config["data"]["target_column"]
        value_counts = df[target_col].value_counts()
        class_imbalance = value_counts.min() / value_counts.max()
        checks.append(class_imbalance > 0.2)  # At least 20% minority class
        logger.info(f"Class distribution:\n{value_counts}")

        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        checks.append(duplicates < 0.05 * len(df))  # Less than 5%
        logger.info(f"Duplicate rows: {duplicates}")

        return all(checks)


loader = DataLoader(config_path="configs/config.yaml")
df = loader.load_raw_data()
is_valid = loader.validate_data(df)

print(f"Data loaded: {df.shape}")
print(f"Is valid: {is_valid}")
