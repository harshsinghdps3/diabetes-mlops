import os
import sys
import logging
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add src to path for module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_config, load_data, preprocess_zeros, validate_columns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train() -> None:
    """
    Main training function. Loads data, preprocesses it, trains a model,
    logs metrics to MLflow, and saves the model.
    """
    try:
        config = load_config()
        
        # Set MLflow experiment
        mlflow.set_experiment("Diabetes_Prediction")

        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(config["model"]["params"])
            mlflow.log_param("test_size", config["data"]["test_size"])
            mlflow.log_param("random_state", config["data"]["random_state"])

            # Load Data
            logger.info("Loading data...")
            try:
                df = load_data(config["data"]["raw_path"])
            except FileNotFoundError:
                logger.error("Error: data/raw/diabetes.csv not found. Please download it first.")
                return

            # Validate Columns
            # Assuming 'Outcome' is the target and others are features.
            # We can check for a few key columns or all expected columns if defined in config.
            # For now, let's ensure 'Outcome' and 'Glucose' exist as a basic check.
            validate_columns(df, ["Outcome", "Glucose"])

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
            logger.info(f"Test data saved to {config['data']['test_path']}")

            # Build Pipeline
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(**config["model"]["params"])),
                ]
            )

            logger.info("Training model...")
            pipeline.fit(X_train, y_train)

            # Initial validation
            train_acc = accuracy_score(y_train, pipeline.predict(X_train))
            logger.info(f"Training Accuracy: {train_acc:.4f}")
            mlflow.log_metric("training_accuracy", train_acc)

            # Persist Model
            save_path = config["model"]["save_path"]
            model_dir = os.path.dirname(save_path)
            os.makedirs(model_dir, exist_ok=True)

            joblib.dump(pipeline, save_path)
            logger.info(f"Model saved to {save_path}")
            
            # Log model to MLflow
            mlflow.sklearn.log_model(pipeline, "model")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train()
