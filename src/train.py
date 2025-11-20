import os
import sys

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add src to path for module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_config, load_data, preprocess_zeros


def train():
    config = load_config()

    # Load Data
    print("Loading data...")
    # Ensure you have the diabetes.csv in data/raw/ before running
    try:
        df = load_data(config["data"]["raw_path"])
    except FileNotFoundError:
        print("Error: data/raw/diabetes.csv not found. Please download it first.")
        return

    # Preprocessing: Handle 0s
    df = preprocess_zeros(df, config["data"]["cols_with_zeros"])

    # Split Data
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # Save processed data for evaluation/testing
    test_df = pd.DataFrame(X_test, columns=X.columns)
    test_df["Outcome"] = y_test
    test_df.to_csv(config["data"]["test_path"], index=False)

    # Build Pipeline
    # 1. Impute missing values (NaNs we created from 0s)
    # 2. Scale features
    # 3. Train Logistic Regression
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**config["model"]["params"])),
        ]
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Initial validation
    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    print(f"Training Accuracy: {train_acc:.4f}")

    # Persist Model
    save_path = config["model"]["save_path"]
    model_dir = os.path.dirname(save_path)
    os.makedirs(model_dir, exist_ok=True)  # Ensure model directory exists

    joblib.dump(pipeline, save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train()
