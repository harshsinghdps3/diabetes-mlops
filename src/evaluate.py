import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import yaml

def evaluate():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Load model and test data
    model = joblib.load(config['model']['save_path'])
    test_df = pd.read_csv(config['data']['test_path'])
    
    X_test = test_df.drop('Outcome', axis=1)
    y_test = test_df['Outcome']
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate()