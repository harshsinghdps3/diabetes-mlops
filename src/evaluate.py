import pandas as pd
import joblib
import logging
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path for module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate() -> None:
    """
    Evaluate the trained model on test data.
    """
    try:
        config = load_config()
        
        # Load model and test data
        model_path = config['model']['save_path']
        test_path = config['data']['test_path']
        
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        logger.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)
        
        X_test = test_df.drop('Outcome', axis=1)
        y_test = test_df['Outcome']
        
        # Predictions
        logger.info("Generating predictions...")
        y_pred = model.predict(X_test)
        
        # Metrics
        logger.info("Evaluation Results:")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    evaluate()